module.exports = {
  "version": "2.0",
  "title": "VisoMaster Experimental",
  "description": "Logiciel puissant et facile à utiliser pour l'échange et le montage vidéo de visages.",
  "icon": "icon.png",
  "menu": [
    {
      "when": "{{!fs.existsSync('app')}}",
      "text": "install",
      "href": "install.json",
      "default": true
    },
    {
      "when": "{{fs.existsSync('app')}}",
      "text": "start",
      "href": "start.json",
      "default": true
    },
    {
      "text": "update",
      "href": "update.json",
      "when": "{{fs.existsSync('app')}}"
    }
  ]
}
