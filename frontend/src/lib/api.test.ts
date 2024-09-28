import { describe, it, expect, vi } from 'vitest'
import { ApiError } from './api'

// Mock the config module
vi.mock('./config', () => ({
  config: {
    apiUrl: 'http://localhost:8000'
  }
}))

describe('api', () => {
  describe('ApiError', () => {
    it('creates ApiError with correct properties', () => {
      const error = new ApiError('Test error', 404, '/test')

      expect(error.message).toBe('Test error')
      expect(error.status).toBe(404)
      expect(error.endpoint).toBe('/test')
      expect(error.name).toBe('ApiError')
      expect(error).toBeInstanceOf(Error)
    })

    it('handles different error scenarios', () => {
      const networkError = new ApiError('Network error', 0, '/api/test')
      const serverError = new ApiError('Server error', 500, '/api/test')

      expect(networkError.status).toBe(0)
      expect(serverError.status).toBe(500)
      expect(networkError.endpoint).toBe('/api/test')
      expect(serverError.endpoint).toBe('/api/test')
    })
  })

  // Note: Full API integration tests would require MSW setup
  // For now, we focus on testing the ApiError class and basic functionality
  // The actual API calls will be tested in component integration tests
})
