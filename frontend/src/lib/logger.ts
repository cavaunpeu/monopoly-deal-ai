// Simple logging utility for frontend
// Can be easily configured to send logs to different destinations

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogConfig {
  level: LogLevel;
  enableConsole: boolean;
  enableRemote?: boolean;
  remoteEndpoint?: string;
}

const config: LogConfig = {
  level: 'info',
  enableConsole: true,
  // Future: enableRemote: false,
  // Future: remoteEndpoint: '/api/logs'
};

const levels: Record<LogLevel, number> = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
};

class Logger {
  private shouldLog(level: LogLevel): boolean {
    return levels[level] >= levels[config.level];
  }

  private formatMessage(level: LogLevel, message: string): string {
    const timestamp = new Date().toISOString();
    return `[${timestamp}] [${level.toUpperCase()}] ${message}`;
  }

  debug(message: string, ...args: unknown[]): void {
    if (this.shouldLog('debug')) {
      if (config.enableConsole) {
        console.log(this.formatMessage('debug', message), ...args);
      }
    }
  }

  info(message: string, ...args: unknown[]): void {
    if (this.shouldLog('info')) {
      if (config.enableConsole) {
        console.info(this.formatMessage('info', message), ...args);
      }
    }
  }

  warn(message: string, ...args: unknown[]): void {
    if (this.shouldLog('warn')) {
      if (config.enableConsole) {
        console.warn(this.formatMessage('warn', message), ...args);
      }
    }
  }

  error(message: string, ...args: unknown[]): void {
    if (this.shouldLog('error')) {
      if (config.enableConsole) {
        console.error(this.formatMessage('error', message), ...args);
      }
    }
  }

  // Configuration methods for future use
  setLevel(level: LogLevel): void {
    config.level = level;
  }

  setConsoleEnabled(enabled: boolean): void {
    config.enableConsole = enabled;
  }
}

export const logger = new Logger();
