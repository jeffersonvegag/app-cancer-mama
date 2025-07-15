# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copiar package.json y package-lock.json
COPY package*.json ./

# Instalar dependencias
RUN npm ci --only=production

# Copiar código fuente
COPY . .

# Build de producción
RUN npm run build

# Production stage
FROM nginx:alpine

# Copiar archivos build a nginx
COPY --from=build /app/build /usr/share/nginx/html

# Copiar configuración nginx (opcional)
COPY nginx.conf /etc/nginx/nginx.conf

# Exponer puerto
EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
