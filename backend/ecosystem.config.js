module.exports = {
  apps: [{
    name: 'sxgjz-backend',
    script: 'server.js',
    cwd: 'D:\\bs\\sxgjz\\backend',
    instances: 1,
    autorestart: true,
    watch: false,
    env: {
      NODE_ENV: 'production',
      PORT: '3000'
    }
  }]
};