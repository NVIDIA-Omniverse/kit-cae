from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "templates/110.1.3/apps/usd_viewer/omni.usd_viewer.kit"


def test_required_extension_placeholders_present():
    in_deps = False
    deps_lines = []
    for raw in KIT.read_text().splitlines():
        line = raw.strip()
        if line == "[dependencies]":
            in_deps = True
            continue
        if in_deps:
            if line.startswith("[") and line.endswith("]"):
                break
            deps_lines.append(line)
    deps_text = "\n".join(deps_lines)
    assert '"{{ extra_extension_name }}"' in deps_text, (
        "Missing required 'extra' extension placeholder in omni.usd_viewer.kit dependencies. "
        "README states the app requires both 'extra' and 'setup' extensions."
    )
    assert '"{{ setup_extension_name }}"' in deps_text, (
        "Missing required 'setup' extension placeholder in omni.usd_viewer.kit dependencies."
    )


if __name__ == "__main__":
    test_required_extension_placeholders_present()
    print("ok")
