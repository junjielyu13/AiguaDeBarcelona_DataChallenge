function prediction(theform) {
  types = theform.flexRadioDefault["value"];
  zona = theform.zonaSelector["value"];

  path = "../model/images/" + types + "/" + zona;

  let comerciallist = [
    "BADALONA",
    "BARCELONA",
    "BEGUES",
    "CASTELLDEFELS",
    "CERDANYOLA",
    "CORNELLA",
    "EL PAPIOL",
    "ESPLUGUES",
    "GAVA",
    "L'HOSPITALET LLOBR",
    "MONTCADA I REIXAC",
    "MONTGAT",
    "PALLEJA",
    "SANT ADRIA",
    "SANT BOI",
    "SANT CLIMENT LLOB",
    "SANT FELIU LL",
    "SANT JOAN DESPI",
    "SANT JUST DESVERN",
    "STA.COLOMA CERVELLO",
    "STA.COLOMA GRAMENET",
    "TORRELLES LLOBREGAT",
    "VILADECANS",
  ];

  let domestico = [
    "BADALONA",
    "BARCELONA",
    "BEGUES",
    "CASTELLDEFELS",
    "CERDANYOLA",
    "CORNELLA",
    "EL PAPIOL",
    "ESPLUGUES",
    "GAVA",
    "L'HOSPITALET LLOBR",
    "MONTCADA I REIXAC",
    "MONTGAT",
    "PALLEJA",
    "SANT ADRIA",
    "SANT BOI",
    "SANT CLIMENT LLOB",
    "SANT FELIU LL",
    "SANT JOAN DESPI",
    "SANT JUST DESVERN",
    "STA.COLOMA CERVELLO",
    "STA.COLOMA GRAMENET",
    "TORRELLES LLOBREGAT",
    "VILADECANS",
  ];

  let industrial = [
    "BADALONA",
    "BARCELONA",
    "BEGUES",
    "CASTELLDEFELS",
    "CERDANYOLA",
    "CORNELLA",
    "EL PAPIOL",
    "ESPLUGUES",
    "GAVA",
    "L'HOSPITALET LLOBR",
    "LA LLAGOSTA",
    "LES BOTIGUES SITGES",
    "MONTCADA I REIXAC",
    "MONTGAT",
    "PALLEJA",
    "SANT ADRIA",
    "SANT BOI",
    "SANT CLIMENT LLOB",
    "SANT CUGAT",
    "SANT FELIU LL",
    "SANT JOAN DESPI",
    "SANT JUST DESVERN",
    "STA.COLOMA CERVELLO",
    "STA.COLOMA GRAMENET",
    "TORRELLES LLOBREGAT",
    "VILADECANS",
  ];

  if (types == "comercial" && comerciallist.includes(zona)) {
    analisi_image = path + "/analisi.png";
    prediction_image = path + "/prediction.png";
  } else if (types == "domestico" && domestico.includes(zona)) {
    analisi_image = path + "/analisi.png";
    prediction_image = path + "/prediction.png";
  } else if (types == "industrial" && industrial.includes(zona)) {
    analisi_image = path + "/analisi.png";
    prediction_image = path + "/prediction.png";
  } else {
    analisi_image = "../static/update/analisi.png";
    prediction_image = "../static/update/prediction.png";
  }

  prediction_box = document.querySelector("#prediction-image");
  prediction_box.src = prediction_image;

  analisi_box = document.querySelector("#analisi-image");
  analisi_box.src = analisi_image;
}

// BADALONA
// BARCELONA
// BEGUES
// CASTELLDEFELS
// CERDANYOLA
// CORNELLA
// EL PAPIOL
// ESPLUGUES
// GAVA
// L'HOSPITALET LLOBR.
// LA LLAGOSTA
// LES BOTIGUES SITGES
// MONTCADA I REIXAC
// MONTGAT
// PALLEJA
// SANT ADRIA
// SANT BOI
// SANT CLIMENT LLOB.
// SANT CUGAT
// SANT FELIU LL.
// SANT JOAN DESPI
// SANT JUST DESVERN
// STA.COLOMA CERVELLO
// STA.COLOMA GRAMENET
// TORRELLES LLOBREGAT
// VILADECANS
