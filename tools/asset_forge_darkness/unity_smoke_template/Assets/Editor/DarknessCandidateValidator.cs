using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace Darkness.EditorTools
{
    /// <summary>
    /// Validates an external Darkness sprite candidate without copying it into the live
    /// game art folders or replacing UnitViewRegistry entries.  The candidate remains a
    /// human-review artifact until the ordinary production importer receives approval.
    /// </summary>
    public static class DarknessCandidateValidator
    {
        [Serializable]
        private sealed class ManifestAction
        {
            public string name;
            public string direction;
            public int frames;
            public float fps;
            public bool loop;
            public string sheet;
            public string sha256;
        }

        [Serializable]
        private sealed class CandidateManifest
        {
            public int schema_version;
            public string asset_id;
            public string status;
            public int cell_width;
            public int cell_height;
            public float pixels_per_unit;
            public float pivot_x;
            public float pivot_y;
            public string source_master;
            public string source_master_sha256;
            public bool automatic_gate_passed;
            public bool human_approval_required;
            public bool human_approved;
            public ManifestAction[] actions;
        }

        [Serializable]
        private sealed class ValidationReport
        {
            public int schema_version = 1;
            public bool passed;
            public string asset_id;
            public string project_kind;
            public string unity_version;
            public string package_path;
            public string candidate_manifest_sha256;
            public string bundle_manifest_sha256;
            public int directional_actions;
            public int decoded_sprites;
            public int animation_clips;
            public bool source_master_hash_verified;
            public bool live_game_assets_modified;
            public string capture_path;
            public string capture_sha256;
            public bool human_approval_required;
            public bool human_approved;
            public string[] checks;
        }

        private sealed class LoadedSheet
        {
            public Texture2D Texture;
            public List<Sprite> Sprites;
        }

        public static void ValidateFromBatch()
        {
            var packagePath = Environment.GetEnvironmentVariable("DARKNESS_CANDIDATE_PACKAGE");
            var outputPath = Environment.GetEnvironmentVariable("DARKNESS_CANDIDATE_OUTPUT");
            var bundleManifestPath = Environment.GetEnvironmentVariable("DARKNESS_BUNDLE_MANIFEST");
            if (string.IsNullOrWhiteSpace(packagePath) || !Directory.Exists(packagePath))
            {
                throw new InvalidOperationException("DARKNESS_CANDIDATE_PACKAGE must name an existing package directory.");
            }

            if (string.IsNullOrWhiteSpace(outputPath))
            {
                throw new InvalidOperationException("DARKNESS_CANDIDATE_OUTPUT must name an evidence directory.");
            }

            if (string.IsNullOrWhiteSpace(bundleManifestPath) || !File.Exists(bundleManifestPath))
            {
                throw new InvalidOperationException("DARKNESS_BUNDLE_MANIFEST must name the immutable bundle manifest.");
            }

            packagePath = Path.GetFullPath(packagePath);
            outputPath = Path.GetFullPath(outputPath);
            bundleManifestPath = Path.GetFullPath(bundleManifestPath);
            Directory.CreateDirectory(outputPath);
            var manifestPath = Path.Combine(packagePath, "candidate_unit_manifest.json");
            if (!File.Exists(manifestPath))
            {
                throw new FileNotFoundException("Candidate manifest is missing.", manifestPath);
            }

            var manifest = JsonUtility.FromJson<CandidateManifest>(File.ReadAllText(manifestPath));
            ValidateManifest(manifest, manifestPath);
            if (Path.IsPathRooted(manifest.source_master))
            {
                throw new InvalidOperationException("Standalone smoke candidates require a package-relative source master.");
            }
            var sourceMasterPath = Path.GetFullPath(Path.Combine(packagePath, manifest.source_master));
            var packagePrefix = packagePath.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
                + Path.DirectorySeparatorChar;
            if (!sourceMasterPath.StartsWith(packagePrefix, StringComparison.OrdinalIgnoreCase)
                || !File.Exists(sourceMasterPath))
            {
                throw new InvalidOperationException("Candidate source master escapes or is missing from the portable package.");
            }
            var sourceHashVerified = string.Equals(
                Sha256(sourceMasterPath),
                manifest.source_master_sha256,
                StringComparison.OrdinalIgnoreCase);
            if (!sourceHashVerified)
            {
                throw new InvalidOperationException("Candidate source-master SHA-256 does not match its manifest.");
            }

            var loaded = new List<LoadedSheet>();
            var clips = new List<AnimationClip>();
            var representative = new Dictionary<string, Sprite>(StringComparer.Ordinal);
            var decodedSprites = 0;
            try
            {
                foreach (var action in manifest.actions)
                {
                    var sheetPath = Path.Combine(packagePath, action.sheet);
                    if (!File.Exists(sheetPath))
                    {
                        throw new FileNotFoundException("Candidate sheet is missing.", sheetPath);
                    }

                    if (!string.Equals(Sha256(sheetPath), action.sha256, StringComparison.OrdinalIgnoreCase))
                    {
                        throw new InvalidOperationException("Candidate sheet hash mismatch: " + action.sheet);
                    }

                    var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, false)
                    {
                        name = Path.GetFileNameWithoutExtension(action.sheet),
                        filterMode = FilterMode.Bilinear
                    };
                    if (!ImageConversion.LoadImage(texture, File.ReadAllBytes(sheetPath), false))
                    {
                        UnityEngine.Object.DestroyImmediate(texture);
                        throw new InvalidOperationException("Unity failed to decode candidate sheet: " + action.sheet);
                    }

                    if (texture.width != manifest.cell_width * action.frames || texture.height != manifest.cell_height)
                    {
                        UnityEngine.Object.DestroyImmediate(texture);
                        throw new InvalidOperationException("Candidate sheet dimensions do not match its frame contract: " + action.sheet);
                    }

                    var sheet = new LoadedSheet { Texture = texture, Sprites = new List<Sprite>() };
                    loaded.Add(sheet);
                    for (var frame = 0; frame < action.frames; frame++)
                    {
                        ValidateFrameAlpha(texture, frame, manifest.cell_width, manifest.cell_height, action.sheet);
                        var sprite = Sprite.Create(
                            texture,
                            new Rect(frame * manifest.cell_width, 0, manifest.cell_width, manifest.cell_height),
                            new Vector2(manifest.pivot_x, manifest.pivot_y),
                            manifest.pixels_per_unit,
                            0,
                            SpriteMeshType.FullRect);
                        sprite.name = string.Format("{0}_{1}_{2:000}", action.name, action.direction, frame);
                        sheet.Sprites.Add(sprite);
                        decodedSprites++;
                    }

                    var stateName = action.name + "_" + action.direction;
                    var clip = BuildClip(stateName, action, sheet.Sprites);
                    clips.Add(clip);
                    if (string.Equals(action.direction, "south", StringComparison.Ordinal))
                    {
                        var index = string.Equals(action.name, "idle", StringComparison.Ordinal)
                            ? 0
                            : string.Equals(action.name, "death", StringComparison.Ordinal)
                                ? sheet.Sprites.Count - 1
                                : sheet.Sprites.Count / 2;
                        representative[action.name] = sheet.Sprites[index];
                    }
                }

                if (representative.Count != 4)
                {
                    throw new InvalidOperationException("Candidate capture requires south-facing idle, walk, attack, and death sprites.");
                }

                var capturePath = Path.Combine(outputPath, "unity_candidate_capture.png");
                RenderCapture(representative, capturePath);
                var report = new ValidationReport
                {
                    passed = true,
                    asset_id = manifest.asset_id,
                    project_kind = "darkness_standalone_unity_smoke",
                    unity_version = Application.unityVersion,
                    package_path = packagePath,
                    candidate_manifest_sha256 = Sha256(manifestPath),
                    bundle_manifest_sha256 = Sha256(bundleManifestPath),
                    directional_actions = manifest.actions.Length,
                    decoded_sprites = decodedSprites,
                    animation_clips = clips.Count,
                    source_master_hash_verified = sourceHashVerified,
                    live_game_assets_modified = false,
                    capture_path = capturePath,
                    capture_sha256 = Sha256(capturePath),
                    human_approval_required = true,
                    human_approved = false,
                    checks = new[]
                    {
                        "manifest_contract",
                        "bundle_manifest_sha256",
                        "portable_source_path",
                        "source_master_sha256",
                        "sheet_sha256",
                        "png_decode",
                        "frame_dimensions",
                        "nonempty_alpha",
                        "no_edge_clipping",
                        "sprite_creation",
                        "animation_clip_creation",
                        "offscreen_candidate_capture"
                    }
                };
                File.WriteAllText(
                    Path.Combine(outputPath, "unity_candidate_validation.json"),
                    JsonUtility.ToJson(report, true));
                Debug.Log("Darkness candidate validation passed: " + JsonUtility.ToJson(report));
            }
            finally
            {
                foreach (var clip in clips)
                {
                    UnityEngine.Object.DestroyImmediate(clip);
                }

                foreach (var sheet in loaded)
                {
                    foreach (var sprite in sheet.Sprites)
                    {
                        UnityEngine.Object.DestroyImmediate(sprite);
                    }
                    UnityEngine.Object.DestroyImmediate(sheet.Texture);
                }
            }
        }

        private static void ValidateManifest(CandidateManifest manifest, string path)
        {
            if (manifest == null
                || manifest.schema_version != 1
                || string.IsNullOrWhiteSpace(manifest.asset_id)
                || !string.Equals(manifest.status, "human_review_candidate", StringComparison.Ordinal)
                || !manifest.automatic_gate_passed
                || !manifest.human_approval_required
                || manifest.human_approved
                || manifest.actions == null
                || manifest.actions.Length != 16
                || manifest.cell_width <= 0
                || manifest.cell_height <= 0
                || manifest.pixels_per_unit <= 0f
                || string.IsNullOrWhiteSpace(manifest.source_master)
                || string.IsNullOrWhiteSpace(manifest.source_master_sha256))
            {
                throw new InvalidOperationException("Invalid Darkness candidate manifest: " + path);
            }

            var expected = new HashSet<string>(StringComparer.Ordinal);
            foreach (var action in new[] { "idle", "walk", "attack", "death" })
            {
                foreach (var direction in new[] { "north", "south", "east", "west" })
                {
                    expected.Add(action + "/" + direction);
                }
            }

            var observed = new HashSet<string>(
                manifest.actions.Select(action => action.name + "/" + action.direction),
                StringComparer.Ordinal);
            if (!expected.SetEquals(observed) || manifest.actions.Any(action => action.frames <= 0 || action.fps <= 0f))
            {
                throw new InvalidOperationException("Candidate does not contain the exact four-clip/four-direction checkpoint.");
            }
        }

        private static void ValidateFrameAlpha(Texture2D texture, int frame, int width, int height, string sheet)
        {
            var colors = texture.GetPixels32();
            var startX = frame * width;
            var visible = false;
            var touchesEdge = false;
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    var alpha = colors[(y * texture.width) + startX + x].a;
                    if (alpha == 0)
                    {
                        continue;
                    }
                    visible = true;
                    if (x <= 1 || y <= 1 || x >= width - 2 || y >= height - 2)
                    {
                        touchesEdge = true;
                    }
                }
            }

            if (!visible || touchesEdge)
            {
                throw new InvalidOperationException(
                    string.Format("Candidate frame alpha gate failed: {0} frame {1}, visible={2}, edge={3}", sheet, frame, visible, touchesEdge));
            }
        }

        private static AnimationClip BuildClip(string stateName, ManifestAction action, IReadOnlyList<Sprite> sprites)
        {
            var clip = new AnimationClip { name = stateName, frameRate = action.fps };
            var keyframes = sprites
                .Select((sprite, index) => new ObjectReferenceKeyframe { time = index / action.fps, value = sprite })
                .ToArray();
            var binding = EditorCurveBinding.PPtrCurve(string.Empty, typeof(SpriteRenderer), "m_Sprite");
            AnimationUtility.SetObjectReferenceCurve(clip, binding, keyframes);
            var settings = AnimationUtility.GetAnimationClipSettings(clip);
            settings.loopTime = action.loop;
            AnimationUtility.SetAnimationClipSettings(clip, settings);
            if (AnimationUtility.GetObjectReferenceCurve(clip, binding).Length != action.frames)
            {
                UnityEngine.Object.DestroyImmediate(clip);
                throw new InvalidOperationException("Unity animation clip lost sprite keyframes: " + stateName);
            }
            return clip;
        }

        private static void RenderCapture(IReadOnlyDictionary<string, Sprite> sprites, string path)
        {
            var root = new GameObject("DarknessCandidateCapture");
            var cameraObject = new GameObject("Camera");
            var renderTexture = new RenderTexture(640, 640, 24, RenderTextureFormat.ARGB32);
            var capture = new Texture2D(640, 640, TextureFormat.RGBA32, false);
            try
            {
                var camera = cameraObject.AddComponent<Camera>();
                camera.orthographic = true;
                camera.orthographicSize = 1.45f;
                camera.transform.position = new Vector3(0f, 0f, -10f);
                camera.backgroundColor = new Color(0.045f, 0.055f, 0.07f, 1f);
                camera.clearFlags = CameraClearFlags.SolidColor;
                camera.targetTexture = renderTexture;
                var layout = new[]
                {
                    new KeyValuePair<string, Vector3>("idle", new Vector3(-0.72f, 0.72f, 0f)),
                    new KeyValuePair<string, Vector3>("walk", new Vector3(0.72f, 0.72f, 0f)),
                    new KeyValuePair<string, Vector3>("attack", new Vector3(-0.72f, -0.72f, 0f)),
                    new KeyValuePair<string, Vector3>("death", new Vector3(0.72f, -0.72f, 0f))
                };
                foreach (var item in layout)
                {
                    var child = new GameObject(item.Key);
                    child.transform.SetParent(root.transform, false);
                    child.transform.position = item.Value;
                    child.AddComponent<SpriteRenderer>().sprite = sprites[item.Key];
                }
                camera.Render();
                var previous = RenderTexture.active;
                RenderTexture.active = renderTexture;
                capture.ReadPixels(new Rect(0, 0, 640, 640), 0, 0);
                capture.Apply();
                RenderTexture.active = previous;
                File.WriteAllBytes(path, capture.EncodeToPNG());
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(capture);
                renderTexture.Release();
                UnityEngine.Object.DestroyImmediate(renderTexture);
                UnityEngine.Object.DestroyImmediate(cameraObject);
                UnityEngine.Object.DestroyImmediate(root);
            }
        }

        private static string Sha256(string path)
        {
            using (var stream = File.OpenRead(path))
            using (var sha = SHA256.Create())
            {
                return string.Concat(sha.ComputeHash(stream).Select(value => value.ToString("x2")));
            }
        }
    }
}
