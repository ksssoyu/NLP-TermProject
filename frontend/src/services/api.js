import axios from 'axios';

const api = axios.create({
  baseURL: 'http://127.0.0.1:8000', // FastAPI 서버 주소에 맞게 수정
});

export default api;
