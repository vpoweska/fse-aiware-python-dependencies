# Helper file to build a docker file based off of our model intuitions
import docker
from time import sleep
import os
from io import BytesIO


class DockerHelper():
    def __init__(self, logging=False, image_name="", dockerfile_name="",
                 container_name="") -> None:
        self.dockerfile_out  = ""
        self.image_name      = image_name
        self.dockerfile_name = dockerfile_name
        self.container_name  = container_name
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException as e:
            if "Permission denied" in str(e):
                print("ERROR: Cannot access Docker socket - Permission Denied")
                raise
            else:
                raise
        self.logging        = logging
        self.previous_error = {"error_message": '', "module": ''}

    def query_docker(self):
        return self.client.api.images()

    def get_project_dir(self, file):
        split_path = file.split('/')
        file_path  = '/'.join(split_path[:-1])
        file_name  = split_path[-1]
        dir_name   = split_path[-2]
        return file_path, dir_name, file_name

    def create_dockerfile(self, llm_out, file):
        project_dir, dir_name, project_file = self.get_project_dir(file)
        self.dockerfile_out = ""
        self.dockerfile_out += f"FROM python:{llm_out['python_version']}\n"
        self.dockerfile_out += f"WORKDIR /app\n"
        self.dockerfile_out += f'RUN ["pip","install","--upgrade","pip"]\n'

        python_modules = llm_out['python_modules']
        if self.logging: print(python_modules)

        for module in python_modules:
            if type(module) == dict:
                name    = module['module']
                version = module['version']
            else:
                name    = module
                version = python_modules[module]

            if type(version) == str:
                # KEY CHANGE: added --trusted-host files.pythonhosted.org
                # and --no-cache-dir to fix hash mismatch errors on old packages
                # (e.g. tensorflow==2.1.0 for Python 2.7)
                self.dockerfile_out += (
                    f'RUN ["pip","install",'
                    f'"--trusted-host","pypi.python.org",'
                    f'"--trusted-host","files.pythonhosted.org",'
                    f'"--no-cache-dir",'
                    f'"--default-timeout=100",'
                    f'"{name}=={version}"]\n'
                )
            else:
                self.dockerfile_out += (
                    f'RUN ["pip","install",'
                    f'"--trusted-host","pypi.python.org",'
                    f'"--trusted-host","files.pythonhosted.org",'
                    f'"--no-cache-dir",'
                    f'"--default-timeout=100",'
                    f'"{name}=={version[0]}"]\n'
                )

        self.dockerfile_out += f"COPY {project_file} /app\n"
        self.dockerfile_out += f'CMD ["python", "/app/{project_file}"]'

        self.image_name      = f"test/pllm:{dir_name}_{llm_out['python_version']}"
        self.container_name  = f"{dir_name}_{llm_out['python_version']}"
        self.dockerfile_name = f"Dockerfile-llm-{llm_out['python_version']}"

        with open(f"{project_dir}/{self.dockerfile_name}", "w") as f:
            f.write(self.dockerfile_out)

    def build_dockerfile(self, path, dockerfile=None):
        if not dockerfile:
            dockerfile = self.dockerfile_name
        error_lines = ""
        project_dir, dir_name, project_file = self.get_project_dir(path)
        for line in self.client.api.build(
            path=project_dir, dockerfile=dockerfile,
            forcerm=True, tag=self.image_name
        ):
            decoded_line = line.decode('utf-8')
            if ('ERROR' in decoded_line
                    or 'Could not fetch URL' in decoded_line
                    or 'errorDetail' in decoded_line):
                error_lines += decoded_line
            if self.logging:
                print(decoded_line)

        if error_lines == "":
            return True, ""
        else:
            return False, error_lines

    def delete_container(self):
        try:
            self.client.containers.get(self.container_name).remove(
                v=True, force=True
            )
        except Exception as e:
            if self.logging: print(e)

    def delete_image(self):
        try:
            self.client.images.remove(image=self.image_name, force=True)
        except Exception as e:
            if self.logging: print(e)

    def run_container_test(self):
        self.delete_container()
        logs = ''
        try:
            self.container = self.client.containers.create(
                self.image_name, name=self.container_name
            )
            self.container.start()
            # Reduced from sleep(10) to sleep(2) — saves 8 seconds per snippet
            sleep(2)
            self.container.reload()
            timeout_count = 0
            while self.container.status == 'running':
                sleep(2)
                self.container.reload()
                timeout_count += 1
                # Safety: don't wait more than 60 seconds for container to finish
                if timeout_count > 30:
                    print("[DockerHelper] Container timeout — forcing stop")
                    self.container.stop()
                    break
            if self.logging: print(self.container.status)
            logs = self.container.logs()
            self.container.remove(v=True, force=True)
            self.container = None
        except docker.errors.ContainerError as e:
            if self.logging: print(e)
            if self.container:
                self.container.reload()
                while self.container.status == 'running':
                    sleep(2)
                    self.container.reload()
                if self.logging: print(self.container.status)
                logs = self.container.logs()
                self.container.remove(v=True, force=True)
                self.container = None
        except Exception as e:
            if self.logging: print(f"run_container_test error: {e}")

        return logs.decode('utf-8') if isinstance(logs, bytes) else (logs or '')


def main():
    dh = DockerHelper(logging=True, image_name="woof:meow",
                      dockerfile_name="", container_name="")
    dh.run_container_test()


if __name__ == "__main__":
    main()
    print("Done")

