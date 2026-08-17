using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

namespace Text2Model.EditorTools
{
    /// <summary>
    /// Small, non-promoting human-review surface for a portable Text2Model candidate.
    /// It reads the immutable sheets beside the standalone project and never imports
    /// them into a game project.
    /// </summary>
    public sealed class Text2ModelCandidateReviewWindow : EditorWindow
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
        }

        [Serializable]
        private sealed class CandidateManifest
        {
            public string asset_id;
            public string display_name;
            public int cell_width;
            public int cell_height;
            public bool human_approval_required;
            public bool human_approved;
            public ManifestAction[] actions;
        }

        private static readonly string[] ActionNames = { "idle", "walk", "attack", "death" };
        private static readonly string[] ActionLabels = { "Idle  [1]", "Walk  [2]", "Attack  [3]", "Death  [4]" };
        private static readonly string[] DirectionNames = { "north", "south", "east", "west" };
        private static readonly string[] DirectionLabels = { "North  [↑]", "South  [↓]", "East  [→]", "West  [←]" };

        private CandidateManifest _manifest;
        private Texture2D _sheet;
        private ManifestAction _current;
        private string _packagePath;
        private string _error;
        private int _actionIndex;
        private int _directionIndex = 1;
        private double _startedAt;
        private double _pausedAt;
        private bool _paused;
        private bool _autoTour = true;

        [MenuItem("Text2Model/Candidate Motion Review", priority = 1)]
        public static void ShowReview()
        {
            var window = GetWindow<Text2ModelCandidateReviewWindow>();
            window.titleContent = new GUIContent("Candidate Review");
            window.minSize = new Vector2(520f, 660f);
            window.Show();
        }

        private void OnEnable()
        {
            _startedAt = EditorApplication.timeSinceStartup;
            EditorApplication.update += OnEditorUpdate;
            LoadCandidate();
        }

        private void OnDisable()
        {
            EditorApplication.update -= OnEditorUpdate;
            DestroySheet();
        }

        private void OnEditorUpdate()
        {
            if (_current == null || _paused)
            {
                return;
            }

            if (_autoTour && Elapsed >= ReviewDuration(_current))
            {
                SelectAction((_actionIndex + 1) % ActionNames.Length);
            }

            Repaint();
        }

        private double Elapsed
        {
            get
            {
                var now = _paused ? _pausedAt : EditorApplication.timeSinceStartup;
                return Math.Max(0.0, now - _startedAt);
            }
        }

        private static string DefaultPackagePath()
        {
            return Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "candidate"));
        }

        private void LoadCandidate()
        {
            try
            {
                _packagePath = Environment.GetEnvironmentVariable("TEXT2MODEL_FORGE_CANDIDATE_PACKAGE");
                if (string.IsNullOrWhiteSpace(_packagePath))
                {
                    _packagePath = DefaultPackagePath();
                }

                _packagePath = Path.GetFullPath(_packagePath);
                var manifestPath = Path.Combine(_packagePath, "candidate_unit_manifest.json");
                if (!File.Exists(manifestPath))
                {
                    throw new FileNotFoundException("The portable candidate manifest was not found.", manifestPath);
                }

                _manifest = JsonUtility.FromJson<CandidateManifest>(File.ReadAllText(manifestPath));
                if (_manifest == null || _manifest.actions == null || _manifest.actions.Length != 16)
                {
                    throw new InvalidOperationException("The candidate must contain sixteen action/direction sheets.");
                }

                if (!_manifest.human_approval_required || _manifest.human_approved)
                {
                    throw new InvalidOperationException("This viewer only opens an unapproved human-review candidate.");
                }

                _error = null;
                SelectAction(_actionIndex);
            }
            catch (Exception exception)
            {
                _error = exception.Message;
                _manifest = null;
                _current = null;
                DestroySheet();
            }
        }

        private void SelectAction(int index)
        {
            _actionIndex = Mathf.Clamp(index, 0, ActionNames.Length - 1);
            SelectCurrentSheet();
        }

        private void SelectDirection(int index)
        {
            _directionIndex = Mathf.Clamp(index, 0, DirectionNames.Length - 1);
            SelectCurrentSheet();
        }

        private void SelectCurrentSheet()
        {
            if (_manifest == null)
            {
                return;
            }

            _current = _manifest.actions.Single(action =>
                string.Equals(action.name, ActionNames[_actionIndex], StringComparison.Ordinal)
                && string.Equals(action.direction, DirectionNames[_directionIndex], StringComparison.Ordinal));
            var sheetPath = Path.Combine(_packagePath, _current.sheet);
            var texture = new Texture2D(2, 2, TextureFormat.RGBA32, false, false)
            {
                name = Path.GetFileNameWithoutExtension(_current.sheet),
                filterMode = FilterMode.Bilinear,
                hideFlags = HideFlags.HideAndDontSave
            };
            if (!ImageConversion.LoadImage(texture, File.ReadAllBytes(sheetPath), false))
            {
                DestroyImmediate(texture);
                throw new InvalidOperationException("Unity could not decode " + _current.sheet + ".");
            }

            if (texture.width != _manifest.cell_width * _current.frames || texture.height != _manifest.cell_height)
            {
                DestroyImmediate(texture);
                throw new InvalidOperationException("Sheet dimensions do not match the manifest: " + _current.sheet);
            }

            DestroySheet();
            _sheet = texture;
            _startedAt = EditorApplication.timeSinceStartup;
            _pausedAt = _startedAt;
            _paused = false;
            Repaint();
        }

        private void DestroySheet()
        {
            if (_sheet != null)
            {
                DestroyImmediate(_sheet);
                _sheet = null;
            }
        }

        private void TogglePause()
        {
            if (_paused)
            {
                _startedAt += EditorApplication.timeSinceStartup - _pausedAt;
                _paused = false;
            }
            else
            {
                _pausedAt = EditorApplication.timeSinceStartup;
                _paused = true;
            }
            Repaint();
        }

        private void Restart()
        {
            _startedAt = EditorApplication.timeSinceStartup;
            _pausedAt = _startedAt;
            _paused = false;
            Repaint();
        }

        private int CurrentFrame()
        {
            if (_current == null)
            {
                return 0;
            }

            var frame = Mathf.FloorToInt((float)(Elapsed * _current.fps));
            return _current.loop ? frame % _current.frames : Mathf.Min(frame, _current.frames - 1);
        }

        private static double ReviewDuration(ManifestAction action)
        {
            return action.loop ? 2.5 : (action.frames / Math.Max(1.0, action.fps)) + 0.8;
        }

        private void HandleShortcuts()
        {
            var currentEvent = Event.current;
            if (currentEvent.type != EventType.KeyDown)
            {
                return;
            }

            var handled = true;
            switch (currentEvent.keyCode)
            {
                case KeyCode.Alpha1: SelectAction(0); break;
                case KeyCode.Alpha2: SelectAction(1); break;
                case KeyCode.Alpha3: SelectAction(2); break;
                case KeyCode.Alpha4: SelectAction(3); break;
                case KeyCode.UpArrow: SelectDirection(0); break;
                case KeyCode.DownArrow: SelectDirection(1); break;
                case KeyCode.RightArrow: SelectDirection(2); break;
                case KeyCode.LeftArrow: SelectDirection(3); break;
                case KeyCode.Space: TogglePause(); break;
                case KeyCode.R: Restart(); break;
                default: handled = false; break;
            }

            if (handled)
            {
                currentEvent.Use();
            }
        }

        private void OnGUI()
        {
            HandleShortcuts();
            EditorGUILayout.Space(8f);
            EditorGUILayout.LabelField("Text2Model Candidate — Human Motion Review", EditorStyles.boldLabel);
            EditorGUILayout.LabelField(
                _manifest == null ? "Portable candidate" : _manifest.display_name,
                EditorStyles.miniLabel);

            if (!string.IsNullOrWhiteSpace(_error))
            {
                EditorGUILayout.HelpBox(_error, MessageType.Error);
                if (GUILayout.Button("Retry loading candidate"))
                {
                    LoadCandidate();
                }
                return;
            }

            var action = GUILayout.Toolbar(_actionIndex, ActionLabels);
            if (action != _actionIndex)
            {
                SelectAction(action);
            }
            var direction = GUILayout.Toolbar(_directionIndex, DirectionLabels);
            if (direction != _directionIndex)
            {
                SelectDirection(direction);
            }

            EditorGUILayout.Space(6f);
            using (new EditorGUILayout.HorizontalScope())
            {
                _autoTour = EditorGUILayout.ToggleLeft("Auto-tour all four motions", _autoTour, GUILayout.Width(190f));
                if (GUILayout.Button(_paused ? "Play  [Space]" : "Pause  [Space]", GUILayout.Width(120f)))
                {
                    TogglePause();
                }
                if (GUILayout.Button("Restart  [R]", GUILayout.Width(110f)))
                {
                    Restart();
                }
            }

            var frame = CurrentFrame();
            EditorGUILayout.LabelField(
                string.Format("{0} / {1}   •   frame {2}/{3}   •   {4:0.#} fps   •   {5}",
                    ActionNames[_actionIndex], DirectionNames[_directionIndex], frame + 1, _current.frames,
                    _current.fps, _current.loop ? "loop" : "one-shot"),
                EditorStyles.centeredGreyMiniLabel);

            var available = Mathf.Max(280f, Mathf.Min(position.width - 24f, position.height - 185f));
            var preview = GUILayoutUtility.GetRect(available, available, GUILayout.ExpandWidth(true));
            var square = Mathf.Min(preview.width, preview.height);
            var drawRect = new Rect(
                preview.x + ((preview.width - square) * 0.5f),
                preview.y + ((preview.height - square) * 0.5f),
                square,
                square);
            EditorGUI.DrawRect(drawRect, new Color(0.035f, 0.045f, 0.06f, 1f));
            if (_sheet != null)
            {
                GUI.DrawTextureWithTexCoords(
                    drawRect,
                    _sheet,
                    new Rect(frame / (float)_current.frames, 0f, 1f / _current.frames, 1f),
                    true);
            }
            Handles.BeginGUI();
            Handles.color = new Color(0.35f, 0.48f, 0.58f, 0.45f);
            Handles.DrawLine(
                new Vector3(drawRect.xMin + 20f, drawRect.yMax - (drawRect.height * 0.18f)),
                new Vector3(drawRect.xMax - 20f, drawRect.yMax - (drawRect.height * 0.18f)));
            Handles.EndGUI();

            EditorGUILayout.HelpBox(
                "Look for foot sliding, weak attack contact, collapsing elbows/knees/hips, silhouette pops, and whether the death reads clearly. This window does not promote the candidate.",
                MessageType.Info);
        }
    }

    [InitializeOnLoad]
    internal static class Text2ModelCandidateReviewStartup
    {
        static Text2ModelCandidateReviewStartup()
        {
            EditorApplication.delayCall += OpenWhenPortableCandidateExists;
        }

        private static void OpenWhenPortableCandidateExists()
        {
            var package = Environment.GetEnvironmentVariable("TEXT2MODEL_FORGE_CANDIDATE_PACKAGE");
            if (string.IsNullOrWhiteSpace(package))
            {
                package = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "..", "candidate"));
            }

            if (File.Exists(Path.Combine(package, "candidate_unit_manifest.json")))
            {
                Text2ModelCandidateReviewWindow.ShowReview();
            }
        }
    }
}
