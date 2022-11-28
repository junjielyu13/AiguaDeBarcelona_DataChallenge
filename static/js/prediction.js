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

  if (types == "comercial") {
    if (comerciallist.includes(zona)) {
      analisi_image = path + "/analisi.png";
      prediction_image = path + "/prediction.png";
    } else {
      analisi_image = "../static/update/grafic_2_1.jpg";
      prediction_image = "../static/update/grafic_2_1.jpg";
    }
  } else if (types == "domestico") {
    if (domestico.includes(zona)) {
      analisi_image = path + "/analisi.png";
      prediction_image = path + "/prediction.png";
    } else {
      analisi_image = "../static/update/grafic_2_1.jpg";
      prediction_image = "../static/update/grafic_2_1.jpg";
    }
  } else if (types == "industrial") {
    if (industrial.includes(zona)) {
      analisi_image = path + "/analisi.png";
      prediction_image = path + "/prediction.png";
    } else {
      analisi_image = "../static/update/grafic_2_1.jpg";
      prediction_image = "../static/update/grafic_2_1.jpg";
    }
  }

  prediction_box = document.querySelector("#prediction-image");
  prediction_box.src = prediction_image;

  analisi_box = document.querySelector("#analisi-image");
  analisi_box.src = analisi_image;
}
