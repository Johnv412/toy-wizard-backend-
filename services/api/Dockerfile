FROM node:20

COPY package*.json ./
RUN npm ci

COPY . .

RUN ls -la

RUN npx esbuild server/index.ts --platform=node --bundle --format=esm --outdir=dist

ENV PORT=8080
EXPOSE 8080

CMD ["npm", "start"]