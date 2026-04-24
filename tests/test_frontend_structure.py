from pathlib import Path
import re
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendStructureTest(unittest.TestCase):
    def setUp(self):
        self.template = (PROJECT_ROOT / "templates" / "index.html").read_text()
        self.styles = (PROJECT_ROOT / "static" / "css" / "styles.css").read_text()
        self.readme = (PROJECT_ROOT / "README.md").read_text()

    def test_redesigned_public_viewer_shell_is_present(self):
        required_shell_classes = [
            "site-hero",
            "hero-content",
            "workflow-panel",
            "source-actions",
            "plot-stage",
        ]

        for class_name in required_shell_classes:
            with self.subTest(class_name=class_name):
                self.assertIn(class_name, self.template)
                self.assertIn(f".{class_name}", self.styles)

    def test_existing_javascript_hooks_are_preserved(self):
        required_ids = [
            "userInput",
            "fetchStatus",
            "loadMpcButton",
            "mpcSpinner",
            "loadMiriadeButton",
            "miriadeSpinner",
            "loadZtfButton",
            "ztfSpinner",
            "asteroidId",
            "plotButton",
            "plotSpinner",
            "plotPhaseButton",
            "plotPhaseSpinner",
            "exportDropdown",
            "exportAll",
            "exportMpc",
            "exportMiriade",
            "exportZtf",
            "plotStatus",
            "plotContainer",
            "plotForm",
        ]

        for element_id in required_ids:
            with self.subTest(element_id=element_id):
                self.assertRegex(self.template, rf'id="{re.escape(element_id)}"')

    def test_generated_hero_asset_replaces_front_image(self):
        self.assertIn('url("../images/generated-asteroid-hero-layout.png")', self.styles)
        self.assertNotIn("front.png", self.styles)
        self.assertTrue((PROJECT_ROOT / "static" / "images" / "generated-asteroid-hero-layout.png").exists())

    def test_theme_uses_blue_surface_tokens(self):
        expected_blue_tokens = {
            "--muted": "#4f7ea8",
            "--panel": "#f8fbff",
            "--line": "#b9d3ec",
            "--paper": "#eaf4ff",
        }

        for token, color in expected_blue_tokens.items():
            with self.subTest(token=token):
                self.assertRegex(self.styles, rf"{re.escape(token)}:\s*{re.escape(color)};")

    def test_plot_stage_frame_fits_plot_container(self):
        self.assertRegex(self.styles, r"\.plot-stage\s*\{[^}]*overflow:\s*hidden;")
        self.assertRegex(self.styles, r"#plotContainer\s*\{[^}]*height:\s*clamp\(600px,\s*68vh,\s*720px\);")
        self.assertRegex(self.styles, r"#plotContainer\s*\{[^}]*overflow:\s*hidden;")
        self.assertRegex(self.styles, r"#plotContainer\s+\.plot-container,\s*#plotContainer\s+\.svg-container,\s*#plotContainer\s+\.main-svg\s*\{[^}]*width:\s*100%\s*!important;")
        self.assertRegex(self.styles, r"#plotContainer\s+\.plot-container,\s*#plotContainer\s+\.svg-container,\s*#plotContainer\s+\.main-svg\s*\{[^}]*height:\s*100%\s*!important;")

    def test_readme_uses_full_page_screenshot(self):
        self.assertIn("static/images/readme-screenshot.png", self.readme)
        self.assertTrue((PROJECT_ROOT / "static" / "images" / "readme-screenshot.png").exists())


if __name__ == "__main__":
    unittest.main()
