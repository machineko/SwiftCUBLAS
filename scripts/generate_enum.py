import re
import os
from pathlib import Path

def read_cublas_enum(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def convert_cpp_enum_to_swift(cpp_enum):
    enums = re.findall(r'typedef enum {([^}]*)}\s*(\w+);', cpp_enum, re.DOTALL)
    swift_enums = []

    for enum_content, enum_name in enums:
        lines = enum_content.split('\n')
        lines_iter = iter(lines)  # Convert list to iterator
        swift_enum_name = re.sub(r'_t$', '', enum_name)  # Remove '_t' suffix
        swift_enum = f"public enum {swift_enum_name}: Int {{\n"
        value_to_case = {}
        for line in lines_iter:
            line = line.strip()
            if not line:
                continue
            comment = ""
            if '/*' in line and '*/' in line:
                # Inline comment
                line, comment = line.split('/*', 1)
                comment = '// ' + comment.replace('*/', '').strip()
            elif '/*' in line:
                # Start of a multi-line comment
                comment = line[line.index('/*'):]
                line = line[:line.index('/*')].strip()
                comment_lines = []
                while '*/' not in comment:
                    next_line = next(lines_iter).strip()
                    comment += '\n' + next_line
                comment = comment.strip()
                comment_lines = comment.split('\n')
                comment = '\n'.join(['// ' + c.replace('/*', '').replace('*/', '').strip() for c in comment_lines])
            if "=" in line:
                parts = line.split('=')
                enum_case = parts[0].strip()
                enum_value = parts[1].split(',')[0].strip()
                swift_case = enum_case.replace(enum_name.upper() + '_', '').lower()
                swift_case = re.sub(r'([a-z])([A-Z])', r'\1_\2', swift_case).lower()
                if enum_value in value_to_case:
                    # Create an alias for the existing case
                    swift_enum += f"    public static var {swift_case}: {swift_enum_name} {{ return .{value_to_case[enum_value]} }} {comment}\n"
                else:
                    swift_enum += f"    case {swift_case} = {enum_value} {comment}\n"
                    value_to_case[enum_value] = swift_case
            elif line.endswith(','):
                enum_case = line.rstrip(',')
                swift_case = enum_case.replace(enum_name.upper() + '_', '').lower()
                swift_case = re.sub(r'([a-z])([A-Z])', r'\1_\2', swift_case).lower()
                swift_enum += f"    case {swift_case} {comment}\n"
        swift_enum += "}\n"
        swift_enums.append(swift_enum)

    return "\n".join(swift_enums)

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def get_cublas_header_path():
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home is None:
        raise Exception('CUDA_HOME environment variable is not set')
    return os.path.join(cuda_home, 'include', 'cublas_api.h')

if __name__ == "__main__":
    cublas_header_path = get_cublas_header_path()
    cublas_enum_content = read_cublas_enum(cublas_header_path)
    swift_enum_content = convert_cpp_enum_to_swift(cublas_enum_content)
    output_file_path = Path("Sources/SwiftCUBLAS/CUBLASTypes/CUBLASEnums.swift")
    write_to_file(output_file_path, swift_enum_content)
    os.system(f"swift-format {output_file_path} -i {output_file_path}")
