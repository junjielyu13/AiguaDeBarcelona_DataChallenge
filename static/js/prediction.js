const sum_diario_comercial = [
  "BADALONA",
  "BARCELONA",
  "BEGUES",
  "CASTELLDEFELS",
  "CERDANYOLA",
  "CORNELLA",
  "EL PAPIOL",
  "ESPLUGUES",
  "GAVA",
  "L'HOSPITALET LLOBR.",
  "MONTCADA I REIXAC",
  "MONTGAT",
  "PALLEJA",
  "SANT ADRIA",
  "SANT BOI",
  "SANT CLIMENT LLOB.",
  "SANT FELIU LL.",
  "SANT JOAN DESPI",
  "SANT JUST DESVERN",
  "STA.COLOMA CERVELLO",
  "STA.COLOMA GRAMENET",
  "TORRELLES LLOBREGAT",
  "VILADECANS",
];

const sum_diario_domestic = [
  "BADALONA",
  "BARCELONA",
  "BEGUES",
  "CASTELLDEFELS",
  "CERDANYOLA",
  "CORNELLA",
  "EL PAPIOL",
  "ESPLUGUES",
  "GAVA",
  "LES BOTIGUES SITGES",
  "L_HOSPITALET LLOBR.",
  "MONTCADA I REIXAC",
  "MONTGAT",
  "PALLEJA",
  "SANT ADRIA",
  "SANT BOI",
  "SANT CLIMENT LLOB.",
  "SANT FELIU LL.",
  "SANT JOAN DESPI",
  "SANT JUST DESVERN",
  "STA.COLOMA CERVELLO",
  "STA.COLOMA GRAMENET",
  "TORRELLES LLOBREGAT",
  "VILADECANS",
];

const sum_diario_industrial = [
  "BADALONA",
  "BARCELONA",
  "BEGUES",
  "CASTELLDEFELS",
  "CERDANYOLA",
  "CORNELLA",
  "EL PAPIOL",
  "ESPLUGUES",
  "GAVA",
  "L'HOSPITALET LLOBR.",
  "LA LLAGOSTA",
  "LES BOTIGUES SITGES",
  "MONTCADA I REIXAC",
  "MONTGAT",
  "PALLEJA",
  "SANT ADRIA",
  "SANT BOI",
  "SANT CLIMENT LLOB.",
  "SANT CUGAT",
  "SANT FELIU LL.",
  "SANT JOAN DESPI",
  "SANT JUST DESVERN",
  "STA.COLOMA CERVELLO",
  "STA.COLOMA GRAMENET",
  "TORRELLES LLOBREGAT",
  "VILADECANS",
];

const sum_mensual_comercial = [
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

const sum_mensual_domestic = [
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

const sum_mensual_industrial = [
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

function prediction(theform) {
  types = theform.flexRadioDefault["value"];
  zona = theform.zonaSelector["value"];

  // path = "../model/modelv4/images/" + types + "/" + zona + "/";
  if (types == "mensual") {
    path = "../model/modelv4/images/mensual";
  } else {
    path = "../model/modelv4/images/error";
  }

  // images:
  prediction_image_comercial = document.querySelector(
    "#prediction-image-comercial"
  );
  analisi_image_comercial = document.querySelector("#analisi-image-comercial");

  prediction_image_domestic = document.querySelector(
    "#prediction-image-domestic"
  );
  analisi_image_domestic = document.querySelector("#analisi-image-domestic");

  prediction_image_industrial = document.querySelector(
    "#prediction-image-industrial"
  );
  analisi_image_industrial = document.querySelector(
    "#analisi-image-industrial"
  );

  // titles:
  analisi_comercial_title = document.querySelector("#analisi-comercial-title");
  prediction_comercial_title = document.querySelector(
    "#prediction-comercial-title"
  );

  analisi_domestic_title = document.querySelector("#analisi-domestic-title");
  prediction_domestic_title = document.querySelector(
    "#prediction-domestic-title"
  );

  analisi_industrial_title = document.querySelector(
    "#analisi-industrial-title"
  );
  prediction_industrial_title = document.querySelector(
    "#prediction-industiral-title"
  );

  analisi_title_area_comercial = document.querySelector("#analisi-title-area");
  analisi_image_area_comercial = document.querySelector("#analisi-image-area");
  analisi_title_area_domestico = document.querySelector(
    "#analisi-title-domestico"
  );
  analisi_image_area_domestico = document.querySelector(
    "#analisi-image-domestico"
  );
  analisi_title_area_industrial = document.querySelector(
    "#analisi-title-industrial"
  );
  analisi_image_area_industrial = document.querySelector(
    "#analisi-image-industrial"
  );

  error_1 = document.querySelector("#error-1");
  error_2 = document.querySelector("#error-2");
  error_3 = document.querySelector("#error-3");

  if (types == "diario") {
    error_1.style.display = "block";
    error_2.style.display = "block";
    error_3.style.display = "block";

    prediction_comercial_title.innerHTML =
      "Predicción del consumo diario comercial";
    prediction_domestic_title.innerHTML =
      "Predicción del consumo diario domestic";
    prediction_industrial_title.innerHTML =
      "Predicción del consumo diario industrial";
    analisi_comercial_title.innerHTML =
      "Analisi de la tendencia del consumo diario comercial";
    analisi_domestic_title.innerHTML =
      "Analisi de la tendencia del consumo diario domestico";
    analisi_industrial_title.innerHTML =
      "Analisi de la tendencia del consumo diario industria";

    if (sum_diario_comercial.includes(zona)) {
      prediction_image_comercial.src =
        path + "/prediction_comercial_" + zona + ".jpeg";
      analisi_image_comercial.src = analisi_image_comercial =
        path + "/param_comercial_" + zona + ".jpeg";
    } else {
      prediction_image_comercial.src = "../static/update/prediction.png";
      analisi_image_comercial.src = "../static/update/analisi.png";
    }

    if (sum_diario_domestic.includes(zona)) {
      prediction_image_domestic.src =
        path + "/prediction_domestic_" + zona + ".jpeg";
      analisi_image_domestic.src = path + "/param_domestic_" + zona + ".jpeg";
    } else {
      prediction_image_domestic.src = "../static/update/prediction.png";
      analisi_image_domestic.src = "../static/update/analisi.png";
    }

    if (sum_diario_industrial.includes(zona)) {
      prediction_image_industrial.src =
        path + "/prediction_industrial_" + zona + ".jpeg";
      analisi_image_industrial.src =
        path + "/param_industrial_" + zona + ".jpeg";
    } else {
      prediction_image_industrial.src = "../static/update/prediction.png";
      analisi_image_industrial.src = "../static/update/analisi.png";
    }
  } else if (types == "mensual") {
    error_1.style.display = "block";
    error_2.style.display = "block";
    error_3.style.display = "block";

    prediction_comercial_title.innerHTML =
      "Predicción del consumo mensual comercial";
    prediction_domestic_title.innerHTML =
      "Predicción del consumo mensual domestic";
    prediction_industrial_title.innerHTML =
      "Predicción del consumo mensual industrial";
    analisi_comercial_title.innerHTML =
      "Analisi de la tendencia del consumo mensual comercial";
    analisi_domestic_title.innerHTML =
      "Analisi de la tendencia del consumo mensual domestico";
    analisi_industrial_title.innerHTML =
      "Analisi de la tendencia del consumo mensual industria";

    if (sum_diario_comercial.includes(zona)) {
      prediction_image_comercial.src =
        path + "/prediction_comercial_" + zona + ".jpeg";
      analisi_image_comercial.src = analisi_image_comercial =
        path + "/param_comercial_" + zona + ".jpeg";
    } else {
      prediction_image_comercial.src = "../static/update/prediction.png";
      analisi_image_comercial.src = "../static/update/analisi.png";
    }

    if (sum_diario_domestic.includes(zona)) {
      prediction_image_domestic.src =
        path + "/prediction_domestic_" + zona + ".jpeg";
      analisi_image_domestic.src = path + "/param_domestic_" + zona + ".jpeg";
    } else {
      prediction_image_domestic.src = "../static/update/prediction.png";
      analisi_image_domestic.src = "../static/update/analisi.png";
    }

    if (sum_diario_industrial.includes(zona)) {
      prediction_image_industrial.src =
        path + "/prediction_industrial_" + zona + ".jpeg";
      analisi_image_industrial.src =
        path + "/param_industrial_" + zona + ".jpeg";
    } else {
      prediction_image_industrial.src = "../static/update/prediction.png";
      analisi_image_industrial.src = "../static/update/analisi.png";
    }
  } else if (types == "error") {
    prediction_comercial_title.innerHTML =
      "Detecció de errores del consumo diario comercial";
    prediction_domestic_title.innerHTML =
      "Detecció de errores del consumo diario domestic";
    prediction_industrial_title.innerHTML =
      "Detecció de errores del consumo diario industrial";

    error_1.style.display = "none";
    error_2.style.display = "none";
    error_3.style.display = "none";

    if (sum_diario_comercial.includes(zona)) {
      prediction_image_comercial.src =
        path + "/error_comercial_" + zona + ".jpeg";
    } else {
      prediction_image_comercial.src = "../static/update/prediction.png";
    }

    if (sum_diario_domestic.includes(zona)) {
      prediction_image_domestic.src =
        path + "/error_domestic_" + zona + ".jpeg";
    } else {
      prediction_image_domestic.src = "../static/update/prediction.png";
    }

    if (sum_diario_industrial.includes(zona)) {
      prediction_image_industrial.src =
        path + "/error_industrial_" + zona + ".jpeg";
    } else {
      prediction_image_industrial.src = "../static/update/prediction.png";
    }
  } else {
    prediction_image_comercial.src = "../static/update/prediction.png";
    analisi_image_comercial.src = "../static/update/analisi.png";
    prediction_image_domestic.src = "../static/update/prediction.png";
    analisi_image_domestic.src = "../static/update/analisi.png";
    prediction_image_industrial.src = "../static/update/prediction.png";
    analisi_image_industrial.src = "../static/update/analisi.png";
  }
}
