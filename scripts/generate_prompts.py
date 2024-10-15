import argparse
from pathlib import Path

import isort


APIS = ['common', 'image_patch', 'scene_patch']

IGNORE_IMPORTS = [
    'from tiny_eqa.agents.image_patch import ImagePatch'
]


def detect_import(line: str) -> bool:
    """ Detects if the line is an import statement. 
    """
    line = line.strip()
    return line.startswith('import ') or line.startswith('from ')


def detect_header(line: str) -> bool:
    """ Detects if the line is a header, defined as beginning a function or class definition.
    """
    line_noindent = line.lstrip()
    indent = len(line) - len(line_noindent)
    return indent == 0 and (line_noindent.startswith('def ') or line_noindent.startswith('class '))


def collect_block(lines: list[str], index) -> tuple[str, int]:
    """ Returns the a block of code starting at the given index as well as the number of lines in the block.
    """
    indent_level = len(lines[index]) - len(lines[index].lstrip())
    block = []
    block.append(lines[index])
    i = index + 1
    while i < len(lines):
        line = lines[i]
        if line.strip() == '' or line.lstrip().startswith('#'):
            block.append(line)
            i += 1
            continue
        indent = len(line) - len(line.lstrip())
        if indent > indent_level:
            block.append(line)
            i += 1
            continue
        break
    
    # remove trailing whitespace
    while block[-1].strip() == '':
        block.pop()
    block[-1] = block[-1].rstrip()

    return ''.join(block), i - index


def collect_api(filename: Path | str) -> tuple[dict, list]:
    """ Collects the imports and headers from the given file.
    """
    imports = set()
    headers = []

    with open(filename, 'r') as f:
        lines = f.readlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if detect_import(line):
            imports.add(line.strip())
            index += 1
        elif detect_header(line):
            block, block_len = collect_block(lines, index)
            headers.append(block)
            index += block_len
        else:
            index += 1
    return imports, headers


if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser(description='Generate prompts for the TinyEQA dataset.')
    parser.add_argument(
        '-i', '--instructions',
        help='Filename of the instructions for the prompt.'
    )
    parser.add_argument(
        '-a', '--apis', nargs='+', choices=APIS, 
        help='List of API files to include. Imports are sorted and headers are concatenated in the provided order'
    )
    parser.add_argument(
        '-o', '--prompt_filename', default='output.txt', 
        help='Output filename for generated prompt.'
    )
    args = parser.parse_args()
    assert len(args.apis) == len(set(args.apis)), 'APIs must be unique.'

    api2filename = lambda name: f'src/tiny_eqa/agents/{name}.py'

    imports_combined = set()
    headers_combined = []
    for api in args.apis:
        imports, headers = collect_api(api2filename(api))
        imports_combined.update(imports)
        headers_combined.extend(headers)
    imports_combined = isort.code('\n'.join(imports_combined)) # sort imports by category and alphabetically
    headers_combined = '\n\n\n'.join(headers_combined)

    with open(args.instructions, 'r') as f:
        instructions = f.read()
    
    content = instructions + '\n\n' + imports_combined + '\n\n' + headers_combined
    with open(args.prompt_filename, 'w') as f:
        f.write(content)

# python -m scripts.generate_prompts -i prompts/viper_eqa_instructions.txt -a common scene_patch image_patch -o prompts/viper_eqa.txt