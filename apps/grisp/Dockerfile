FROM node:22-alpine AS build
WORKDIR /app
ARG API_BASE="/api"
ARG COPYRIGHT="University of Freiburg"
ARG MATERIAL_URL="grisp"

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN API_BASE=$API_BASE COPYRIGHT=$COPYRIGHT MATERIAL_URL=$MATERIAL_URL npm run build

FROM nginx:1.27-alpine-slim
RUN rm -rf /usr/share/nginx/html/*
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
