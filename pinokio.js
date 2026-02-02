module.exports = {
  "version": "2.0",
  "title": "VisoMaster Experimental",
  "description": "Logiciel puissant et facile à utiliser pour l'échange et le montage vidéo de visages.",
  "icon": "icon.png",
  "menu": async (kernel, info) => {
    let installed = info.exists("app/env")
    if (installed) {
      return [{
        default: true,
        text: "start",
        href: "start.json",
      }, {
        text: "install",
        href: "install.json"
      }]
    } else {
      return [{
        text: "start",
        href: "start.json",
      }, {
        default: true,
        text: "install",
        href: "install.json"
      }]
    }
  }
}
