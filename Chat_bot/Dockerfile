# Stage 1: Build the React application
FROM node:18-alpine as build-stage

WORKDIR /app

ARG VITE_APP_API_URL_BUILD

ENV VITE_APP_API_URL=${VITE_APP_API_URL_BUILD}

COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: Serve the React application with Nginx
FROM nginx:stable-alpine as production-stage

COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]