import { http, HttpResponse } from 'msw'
import { mockGameState, mockGameConfig } from '../fixtures/gameState'

export const handlers = [
  // Create game
  http.post('/game/', () => {
    return HttpResponse.json({ game_id: 'test-game-123' })
  }),

  // Get game state
  http.get('/game/:gameId/state', ({ params }) => {
    const { gameId } = params
    if (gameId === 'test-game-123') {
      return HttpResponse.json(mockGameState)
    }
    return new HttpResponse(null, { status: 404 })
  }),

  // Get default game config
  http.get('/game/config', () => {
    return HttpResponse.json(mockGameConfig)
  }),

  // Get models
  http.get('/game/models', () => {
    return HttpResponse.json({
      models: {
        'cfr-medium': {
          name: 'CFR Medium',
          description: 'Counterfactual Regret Minimization model',
          checkpoint_path: 'gs://monopoly-deal-agent/checkpoints/local/game_idx_2050.json'
        }
      }
    })
  }),

  // Take action
  http.post('/game/:gameId/step', ({ params }) => {
    const { gameId } = params
    if (gameId === 'test-game-123') {
      return HttpResponse.json({ status: 'ok' })
    }
    return new HttpResponse(null, { status: 404 })
  }),

  // Get AI action
  http.get('/game/:gameId/ai_action', ({ params }) => {
    const { gameId } = params
    if (gameId === 'test-game-123') {
      return HttpResponse.json({
        id: 9999,
        is_response: false,
        card: null,
        src: null,
        dst: null
      })
    }
    return new HttpResponse(null, { status: 404 })
  }),

  // Get selection info
  http.get('/game/:gameId/selection_info', ({ params }) => {
    const { gameId } = params
    if (gameId === 'test-game-123') {
      return HttpResponse.json({
        model_type: 'cfr-medium',
        state_key: 'abc123',
        policy: { 'action_1': 0.7, 'action_2': 0.3 },
        update_counts: { 'action_1': 150, 'action_2': 50 }
      })
    }
    return new HttpResponse(null, { status: 404 })
  }),

  // Error scenarios
  http.get('/game/invalid-game/state', () => {
    return new HttpResponse(null, { status: 404 })
  }),

  http.post('/game/invalid-game/step', () => {
    return new HttpResponse(null, { status: 404 })
  }),

  // Network error simulation
  http.get('/game/network-error/state', () => {
    return HttpResponse.error()
  })
]
