FROM node:18-alpine
WORKDIR /app
COPY package.json proxy.js ./
EXPOSE 4000
ENV OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen2.5:7b \
    PROXY_PORT=4000
CMD ["node", "proxy.js"]
