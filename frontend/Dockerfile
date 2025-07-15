FROM node:18-alpine

WORKDIR /app

# Copiar package.json
COPY package.json ./

# Instalar dependencias (usa npm install en lugar de npm ci)
RUN npm install

# Copiar código fuente
COPY . .

# Build de producción
RUN npm run build

# Instalar serve para servir la aplicación
RUN npm install -g serve

# Exponer puerto
EXPOSE 3000

# Servir la aplicación
CMD ["serve", "-s", "build", "-l", "3000"]
