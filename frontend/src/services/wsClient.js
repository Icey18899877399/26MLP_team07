// Compatibility export for imported stores; the transport is HTTP.
import { HttpClient } from './httpClient.js'
import { CONFIG } from '../config'
export const wsClient = new HttpClient({ defaultUrl: CONFIG.defaultWsUrl })
