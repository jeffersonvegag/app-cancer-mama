FROM node:18-alpine

WORKDIR /app

# Copiar package.json
COPY package*.json ./

# Instalar dependencias
RUN npm ci --only=production

# Copiar código fuente
COPY . .

# Build de producción
RUN npm run build

# Instalar serve globalmente
RUN npm install -g serve

# Exponer puerto
EXPOSE 3000

# Servir build de producción
CMD ["serve", "-s", "build", "-l", "3000"]
