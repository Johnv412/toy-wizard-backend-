FROM node:20

ARG NODE_ENV=production
ENV NODE_ENV=$NODE_ENV

COPY package*.json ./
RUN npm ci

COPY . .

RUN npm run build

ENV PORT=8080
EXPOSE 8080

CMD ["npm", "start"]