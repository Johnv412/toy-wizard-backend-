import express, { type Request, Response, NextFunction } from "express";
import cors from "cors";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import { serverConfig } from "./config";
import { registerRoutes } from "./routes";
import { setupVite, serveStatic, log } from "./vite";
// import { serverConfig } from "./config";

const app = express();


// Enable CORS with allowed origins from config
app.use(cors({
  origin: serverConfig.allowedOrigins,
  credentials: true
}));

// Add security middleware
app.use(helmet());
app.use(rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
}));

app.use(express.json());
app.use(express.urlencoded({ extended: false }));

app.use((req, res, next) => {
  const start = Date.now();
  const path = req.path;
  let capturedJsonResponse: Record<string, any> | undefined = undefined;

  const originalResJson = res.json;
  res.json = function (bodyJson, ...args) {
    capturedJsonResponse = bodyJson;
    return originalResJson.apply(res, [bodyJson, ...args]);
  };

  res.on("finish", () => {
    const duration = Date.now() - start;
    if (path.startsWith("/api")) {
      let logLine = `${req.method} ${path} ${res.statusCode} in ${duration}ms`;
      if (capturedJsonResponse) {
        logLine += ` :: ${JSON.stringify(capturedJsonResponse)}`;
      }

      if (logLine.length > 80) {
        logLine = logLine.slice(0, 79) + "…";
      }

      // Only log in development
      if (process.env.NODE_ENV !== "production") {
        log(logLine);
      }
    }
  });

  next();
});

(async () => {
  const server = await registerRoutes(app);

  app.use((err: any, _req: Request, res: Response, _next: NextFunction) => {
    const status = err.status || err.statusCode || 500;
    const message = err.message || "Internal Server Error";

    res.status(status).json({ message });
    // Only log errors in development
    if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.error(err);
    }
    throw err;
  });

  // IMPORTANT: Enforce HTTPS in production (if not handled by platform)
  // app.use((req, res, next) => {
  //   if (process.env.NODE_ENV === "production" && req.headers["x-forwarded-proto"] !== "https") {
  //     return res.redirect("https://" + req.headers.host + req.url);
  //   }
  //   next();
  // });

  // importantly only setup vite in development and after
  // setting up all the other routes so the catch-all route
  // doesn't interfere with the other routes
  if (app.get("env") === "development") {
    await setupVite(app, server);
  } else {
    serveStatic(app);
  }

  // ALWAYS serve the app on the port specified in the environment variable PORT
  // Other ports are firewalled. Default to 5000 if not specified.
  // this serves both the API and the client.
  // It is the only port that is not firewalled.
  const port = parseInt(process.env.PORT || '5000', 10);
  server.listen(port, "0.0.0.0", () => {
    log(`serving on port ${port}`);
  });
})();