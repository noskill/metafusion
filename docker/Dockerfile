FROM nvidia/cuda:12.0.0-devel-ubuntu22.04
RUN apt-get update
RUN apt-get install -y curl libicu70 python3
# needed by opencv
RUN apt-get -y install ffmpeg libsm6 libxext6
RUN useradd -ms /bin/bash tester
user tester
workdir /tester
RUN mkdir actions-runner && cd actions-runner
RUN curl -o actions-runner-linux-x64-2.317.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.317.0/actions-runner-linux-x64-2.317.0.tar.gz
RUN echo "9e883d210df8c6028aff475475a457d380353f9d01877d51cc01a17b2a91161d  actions-runner-linux-x64-2.317.0.tar.gz" | shasum -a 256 -c
RUN tar xzf ./actions-runner-linux-x64-2.317.0.tar.gz
# GITHUB TOKEN
ARG token=vasya
RUN ./config.sh --url https://github.com/singnet/metafusion --token $token
entrypoint /bin/sh
