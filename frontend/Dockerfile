# stage 1: build
FROM node:18-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json* yarn.lock* /app/
# prefer pnpm / yarn if used — adjust accordingly
RUN npm ci --prefer-offline --no-audit --progress=false
COPY . /app
RUN npm run build

# stage 2: nginx static server
FROM nginx:1.25-alpine
COPY --from=builder /app/dist /usr/share/nginx/html
# custom nginx conf to support SPA routing
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]