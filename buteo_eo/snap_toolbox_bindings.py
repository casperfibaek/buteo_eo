import os

def find_gpt(test_gpt_path):
    gpt = os.path.realpath(os.path.abspath(os.path.expanduser(test_gpt_path)))
    if not os.path.exists(gpt):
        possible_locations = [
            "~/esa_snap/bin/gpt",
            "~/snap/bin/gpt",
            "/opt/esa_snap/bin/gpt",
            "/opt/snap/bin/gpt",
            "C:/Program Files/snap/bin/gpt.exe",
            '"C:/Program Files/snap/bin/gpt.exe"',
        ]

        for loc in possible_locations:
            gpt = os.path.realpath(os.path.abspath(os.path.expanduser(loc)))
            if os.path.exists(gpt):
                return gpt
        
        assert os.path.exists(gpt), "Graph processing tool not found."

    else:
        return gpt

def run_gpt(gpt_path, graph, args_as_dict):
    xmlfile = os.path.join(os.path.dirname(__file__), f"./graphs/{graph}")

    

    os.system(" ".join([gpt_path] + args))
