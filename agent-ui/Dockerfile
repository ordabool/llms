FROM node:18-alpine

WORKDIR /app

RUN apk add --no-cache git
RUN npx create-agent-ui@latest

WORKDIR /app/agent-ui

RUN npm install

# CMD ["node"]
CMD ["npm", "run", "dev"]
