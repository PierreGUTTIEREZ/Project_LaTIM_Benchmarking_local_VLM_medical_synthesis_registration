// Diaporama de restitution du volet Synthèse.
// Prérequis : lancer d'abord `python make_figures.py` (écrit slides/data.json et figures/*.pdf),
// puis rendre les images :  for f in banc_essai surrogates distributions_val_gray scores_generateurs;
//   do pdftoppm -png -r 300 -singlefile ../figures/$f.pdf img/$f; done
// Puis : node make_slides.js   → ../restitution.pptx
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const JSZip = require(require.resolve("jszip", { paths: [path.dirname(require.resolve("pptxgenjs"))] }));
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

const SKILL = process.env.PPTX_SKILL_DIR;
const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));

const HERE = __dirname;
const OUT = path.join(HERE, "..", "restitution.pptx");
const D = JSON.parse(fs.readFileSync(path.join(HERE, "data.json"), "utf8"));
const IMG = (n) => path.join(HERE, "img", `${n}.png`);

const THEME = {
  name: "LaTIM Synthese",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "0E1B24", lt1: "FFFFFF", dk2: "12384A", lt2: "EEF2F4",
    accent1: "1F5FAD", accent2: "C0700F", accent3: "1B9E6E", accent4: "D64541",
    accent5: "6250D6", accent6: "6B7A83", hlink: "1F5FAD", folHlink: "6250D6",
  },
};
const HEX = THEME.colors;
const GRIDHEX = "D9DEE1";

// Nombres à la française : virgule décimale, vrai signe moins.
const fr = (v, d = 2) => (v < 0 ? "−" : "") + Math.abs(v).toFixed(d).replace(".", ",");

async function icon(Comp, hex) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13,333 × 7,5 pouces
  pres.title = "Des VLM locaux peuvent-ils juger la qualité de CT synthétiques ?";
  pres.subject = "Restitution du volet Synthèse, sujet A";
  pres.author = "Équipe du sujet A";
  pres.company = "IMT Atlantique - LaTIM";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
  const C = pres.SchemeColor;
  const W = 13.333, M = 0.6, CW = W - 2 * M;

  // --- Layouts -------------------------------------------------------------------------------
  const footer = { text: { text: "VLM locaux juges de CT synthétiques · Sujet A, volet Synthèse",
    options: { x: M, y: 7.0, w: 8, h: 0.3, fontSize: 10, color: C.accent6, margin: 0 } } };
  pres.defineSlideMaster({
    title: "TITRE_SOMBRE",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.7, w: 7.9, h: 2.3, fontSize: 40, bold: true,
        color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.25, w: 7.9, h: 0.9, fontSize: 20,
        color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
      { placeholder: { options: { name: "auteurs", type: "body", x: 0.8, y: 5.7, w: 7.9, h: 0.9, fontSize: 14,
        color: C.background2, valign: "top", align: "left", margin: 0 }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENU",
    background: { color: C.background1 },
    margin: [0.5, M, 0.6, M],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: CW, h: 0.85, fontSize: 34, bold: true,
        color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
      footer,
    ],
    slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CONCLUSION_SOMBRE",
    background: { color: C.text2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: M, y: 0.4, w: CW, h: 0.85, fontSize: 34, bold: true,
        color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    ],
    slideNumber: { x: W - M - 0.6, y: 7.0, w: 0.6, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
  });

  // --- Briques réutilisables -----------------------------------------------------------------
  const ic = {};
  for (const [k, comp, hex] of [
    ["brain", fa.FaBrain, HEX.lt1], ["radiation", fa.FaRadiation, HEX.lt1], ["question", fa.FaQuestion, HEX.lt1],
    ["robot", fa.FaRobot, HEX.lt1], ["chart", fa.FaChartLine, HEX.lt1], ["layers", fa.FaLayerGroup, HEX.lt1],
    ["sort", fa.FaSortAmountDown, HEX.lt1], ["filter", fa.FaFilter, HEX.lt1], ["warn", fa.FaExclamationTriangle, HEX.lt1],
    ["check", fa.FaCheck, HEX.lt1], ["times", fa.FaTimes, HEX.lt1], ["bulb", fa.FaLightbulb, HEX.lt1],
    ["redo", fa.FaRedo, HEX.lt1], ["eye", fa.FaEye, HEX.lt1], ["balance", fa.FaBalanceScale, HEX.lt1],
  ]) ic[k] = await icon(comp, hex);

  let n = 0;
  const id = (s) => `${s}_${++n}`;
  function badge(slide, x, y, d, key, fill) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { type: "none" },
      objectName: id("pastille") });
    const p = d * 0.25;
    slide.addImage({ data: ic[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: id("icone") });
  }
  function numberBadge(slide, x, y, d, num, fill) {
    slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill || C.accent1 }, line: { type: "none" },
      objectName: id("numero") });
    slide.addText(String(num), { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: 16, bold: true,
      color: C.background1, margin: 0, isTextBox: true, objectName: id("chiffre") });
  }
  function card(slide, x, y, w, h, fill) {
    slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || C.background2 },
      line: { type: "none" }, objectName: id("carte") });
  }
  function text(slide, t, o) {
    slide.addText(t, Object.assign({ margin: 0, isTextBox: true, valign: "top", fontSize: 15, color: C.text1,
      objectName: id("texte") }, o));
  }
  function stat(slide, x, y, w, big, label, color) {
    text(slide, big, { x, y, w, h: 0.75, fontSize: 36, bold: true, color: color || C.accent1, fontFace: THEME.headFontFace,
      valign: "bottom" });
    text(slide, label, { x, y: y + 0.8, w, h: 0.75, fontSize: 14, color: C.text1 });
  }
  const chartText = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    dataLabelFontFace: "+mn-lt", titleFontFace: "+mn-lt", catAxisLabelColor: HEX.dk1, valAxisLabelColor: HEX.accent6,
    catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, legendFontSize: 12, dataLabelFontSize: 11,
    valGridLine: { color: GRIDHEX, size: 0.75 }, catGridLine: { style: "none" } };
  const note = (s, t) => s.addNotes(t);

  // ===== Section 1 : contexte ===================================================================
  pres.addSection({ title: "Contexte" });
  let s = pres.addSlide({ masterName: "TITRE_SOMBRE", sectionTitle: "Contexte" });
  s.addText("Des VLM locaux peuvent-ils juger la qualité de CT synthétiques ?", { placeholder: "title" });
  s.addText("Banc d'essai sur SynthRAD2023 (IRM → CT, cerveau) · 10 VLM · 20\u00A0396 jugements", { placeholder: "body" });
  s.addText([
    { text: "Pierre Guttierez et l'équipe du sujet A", options: { bold: true, breakLine: true } },
    { text: "Encadrant : Vincent Jaouen (IMT Atlantique, LaTIM) · novembre 2026" },
  ], { placeholder: "auteurs" });
  // motif : coupe axiale stylisée (anneaux concentriques)
  for (const [r, t] of [[2.1, 55], [1.55, 35], [1.0, 15]]) {
    s.addShape(pres.shapes.OVAL, { x: 11.2 - r, y: 3.75 - r * 1.12, w: 2 * r, h: 2 * r * 1.12,
      fill: { color: C.accent1, transparency: t + 30 }, line: { color: C.background2, width: 1, transparency: 40 },
      objectName: id("motif") });
  }
  badge(s, 10.7, 3.25, 1.0, "brain", C.accent2);
  note(s, "Objet : savoir si un modèle vision–langage exécuté en local peut noter un CT synthétique sans voir le vrai CT. " +
    "Tout a été fait sur la station de l'équipe (RTX 5090), sans aucune donnée envoyée à l'extérieur.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Contexte" });
  s.addText("Le verrou : juger un sCT sans le vrai CT", { placeholder: "title" });
  const ctx = [
    ["brain", "MRI-only", "L'IRM sert à délinéer la tumeur ; un CT synthétique (sCT) remplace le scanner de planification."],
    ["radiation", "La dose dépend du sCT", "Une erreur de densité dans l'os ou les cavités d'air devient une erreur de dose."],
    ["question", "Pas de vérité en routine", "Le vrai CT n'existe pas : impossible de calculer la MAE ou le gamma du sCT produit."],
  ];
  ctx.forEach(([k, h, b], i) => {
    const x = M + i * (3.86 + 0.27);
    card(s, x, 1.55, 3.86, 2.55);
    badge(s, x + 0.3, 1.8, 0.62, k);
    text(s, h, { x: x + 1.08, y: 1.8, w: 2.6, h: 0.62, fontSize: 19, bold: true, color: C.text2, valign: "middle" });
    text(s, b, { x: x + 0.3, y: 2.6, w: 3.3, h: 1.35, fontSize: 15 });
  });
  card(s, M, 4.45, CW, 1.85, "DCE6F0");
  text(s, "Question de recherche (vade-mecum)", { x: M + 0.35, y: 4.65, w: 8, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  text(s, [
    { text: "Un VLM local, sans entraînement spécifique et sans voir la vérité terrain, peut-il noter la qualité d'un sCT ? ",
      options: { breakLine: true } },
    { text: "Lequel juge le mieux, à quel coût ? Un modèle médical (MedGemma) fait-il mieux qu'un généraliste plus gros ?" },
  ], { x: M + 0.35, y: 5.1, w: CW - 0.7, h: 1.05, fontSize: 18, color: C.text2 });
  note(s, "En MRI-only, le sCT est utilisé pour calculer la dose. Le contrôler suppose normalement le vrai CT, " +
    "qui est justement absent. Un juge automatique sans vérité terrain permettrait de contrôler chaque sCT, " +
    "de comparer des modèles de synthèse sur des données non annotées et de filtrer des données synthétiques.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Contexte" });
  s.addText("La chaîne expérimentale", { placeholder: "title" });
  const boxes = [
    ["SynthRAD2023", "180 patients, 3 centres"], ["10 générateurs", "U-Net ×7 checkpoints\nFedSynthCT ×3"],
    ["270 sCT", "par partie (27 patients)"], ["810 images", "3 coupes axiales par sCT"],
    ["VLM local", "Ollama, JSON imposé"], ["Note 0–100", "+ justification"],
  ];
  const bw = 1.78, gap = (CW - 6 * bw) / 5, by = 1.75, bh = 1.35;
  boxes.forEach(([h, b], i) => {
    const x = M + i * (bw + gap);
    const vlm = i === 4;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: by, w: bw, h: bh, rectRadius: 0.08,
      fill: { color: vlm ? C.accent1 : C.background2 }, line: { type: "none" }, objectName: id("etape") });
    text(s, [{ text: h, options: { bold: true, fontSize: 16, breakLine: true } }, { text: b, options: { fontSize: 12 } }],
      { x: x + 0.1, y: by + 0.12, w: bw - 0.2, h: bh - 0.24, align: "center", valign: "middle",
        color: vlm ? C.background1 : C.text2 });
    if (i < 5) s.addShape(pres.shapes.LINE, { x: x + bw + 0.05, y: by + bh / 2, w: gap - 0.1, h: 0,
      line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: id("fleche") });
  });
  // chemin de la vérité terrain (ambre, tirets)
  const gx0 = M + bw / 2, gy = 4.05, mx = M + 2 * (bw + gap), ax = M + 5 * (bw + gap);
  s.addShape(pres.shapes.LINE, { x: gx0, y: by + bh, w: 0, h: gy - by - bh + 0.45,
    line: { color: C.accent2, width: 2, dashType: "dash" }, objectName: id("verite") });
  s.addShape(pres.shapes.LINE, { x: gx0, y: gy + 0.45, w: mx - gx0, h: 0,
    line: { color: C.accent2, width: 2, dashType: "dash", endArrowType: "triangle" }, objectName: id("verite") });
  text(s, "vrai CT", { x: gx0 + 0.15, y: gy + 0.5, w: 1.6, h: 0.3, fontSize: 12, color: C.accent2, bold: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: mx, y: gy, w: bw * 2 + gap, h: 0.9, rectRadius: 0.08,
    fill: { color: C.background1 }, line: { color: C.accent2, width: 1.5 }, objectName: id("metriques") });
  text(s, [{ text: "Métriques de référence", options: { bold: true, breakLine: true } },
    { text: "MAE, PSNR, SSIM · gamma, DVH", options: { fontSize: 12 } }],
    { x: mx + 0.1, y: gy + 0.08, w: bw * 2 + gap - 0.2, h: 0.74, align: "center", valign: "middle", fontSize: 15, color: C.text2 });
  s.addShape(pres.shapes.LINE, { x: mx + 0.6, y: by + bh, w: 0, h: gy - by - bh,
    line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: id("fleche") });
  s.addShape(pres.shapes.LINE, { x: mx + bw * 2 + gap, y: gy + 0.45, w: ax - (mx + bw * 2 + gap), h: 0,
    line: { color: C.accent2, width: 2, dashType: "dash", endArrowType: "triangle" }, objectName: id("verite") });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: ax, y: gy, w: bw, h: 0.9, rectRadius: 0.08,
    fill: { color: C.text2 }, line: { type: "none" }, objectName: id("analyse") });
  text(s, "Analyse", { x: ax, y: gy, w: bw, h: 0.9, align: "center", valign: "middle", bold: true, fontSize: 16, color: C.background1 });
  s.addShape(pres.shapes.LINE, { x: ax + bw / 2, y: by + bh, w: 0, h: gy - by - bh,
    line: { color: C.accent6, width: 1.5, endArrowType: "triangle" }, objectName: id("fleche") });
  text(s, "jamais vues par le VLM", { x: mx + bw * 2 + gap + 0.05, y: gy + 0.08, w: ax - (mx + bw * 2 + gap) - 0.1, h: 0.3,
    fontSize: 11, color: C.accent2, align: "center" });
  const rules = ["Découpage par patient et par centre : 126 / 27 / 27", "Tous les réglages sur la validation ; test utilisé une fois",
    "Le VLM ne reçoit qu'une image et une consigne"];
  rules.forEach((r, i) => {
    const x = M + i * (3.86 + 0.27);
    badge(s, x, 5.55, 0.5, ["layers", "redo", "eye"][i], C.text2);
    text(s, r, { x: x + 0.65, y: 5.5, w: 3.15, h: 0.75, fontSize: 14, valign: "middle" });
  });
  note(s, "Le chemin orange porte la vérité terrain. Le vrai CT ne sert qu'à calculer les métriques de référence, " +
    "qui vont directement à l'analyse. Le VLM ne voit jamais ni le vrai CT, ni les métriques, ni le nom du générateur. " +
    "Le fond hors du corps est remis à l'air pour tous les sCT, sinon le VLM pourrait reconnaître un générateur à son fond.");

  // ===== Section 2 : banc d'essai ===============================================================
  pres.addSection({ title: "Banc d'essai" });
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Banc d'essai" });
  s.addText("Un banc d'essai du très mauvais au bon", { placeholder: "title" });
  s.addImage({ path: IMG("banc_essai"), x: M, y: 1.5, w: 8.3, h: 8.3 / 2.495, objectName: id("figure_banc"),
    altText: "Boîtes à moustaches : MAE du volume et taux gamma protons pour les dix générateurs (test)." });
  text(s, "Test, 27 patients. (a) MAE du volume ; (b) gamma 2 %/2 mm en protons. Bleu : U-Net ; orange : FedSynthCT-Brain.",
    { x: M, y: 4.9, w: 8.3, h: 0.5, fontSize: 11, color: C.accent6 });
  stat(s, 9.45, 1.5, 3.3, "249 → 98 HU", "MAE du U-Net, de l'époque 1 à l'époque 100");
  stat(s, 9.45, 3.15, 3.3, "35 %", "des coupes ont une MAE < 100 HU (seuil « acceptable »)");
  stat(s, 9.45, 4.8, 3.3, "≈ 97 %", "gamma protons des modèles pré-entraînés, malgré une MAE de 136 HU", C.accent2);
  card(s, M, 5.65, 8.3, 1.0);
  text(s, "Pas de sCT « état de l'art » (MAE 50–60 HU) : les juges ne sont pas testés entre « bon » et « très bon ».",
    { x: M + 0.25, y: 5.75, w: 7.8, h: 0.8, fontSize: 14, valign: "middle" });
  note(s, "Sept checkpoints du U-Net (époques 1 à 100) et trois modèles publiés (Li, Fu, Spadea). " +
    "Chaque partie compte 270 sCT. Les pré-entraînés ont une MAE plus élevée que le U-Net à 50 ou 100 époques " +
    "mais une dose protons aussi bonne : la MAE ne résume pas l'utilité dosimétrique.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Banc d'essai" });
  s.addText("Les métriques de référence divergent", { placeholder: "title" });
  s.addImage({ path: IMG("surrogates"), x: M, y: 1.45, w: 5.2 * 1.271, h: 5.2, objectName: id("figure_surrogates"),
    altText: "Matrice des corrélations de Spearman orientées entre les métriques image et dose." });
  const sx = 7.75;
  stat(s, sx, 1.5, 4.9, "0,93", "MAE ↔ SSIM : les métriques image sont presque redondantes");
  stat(s, sx, 3.1, 4.9, "0,43 · 0,62", "MAE ↔ gamma photons · protons : lien partiel avec la dose");
  stat(s, sx, 4.7, 4.9, "0,33", "gamma photons ↔ gamma protons : les deux doses ne racontent pas la même chose", C.accent2);
  text(s, "540 sCT (validation et test). Plan optimisé sur le vrai CT puis recalculé à l'identique sur chaque sCT.",
    { x: sx, y: 6.3, w: 4.9, h: 0.5, fontSize: 11, color: C.accent6 });
  note(s, "Réponse à la question de brainstorming sur les surrogates. La MAE dit si le sCT ressemble au CT, " +
    "pas s'il donne la bonne dose. En photons, la dose est peu sensible aux erreurs de HU ; en protons, beaucoup plus. " +
    "La cible de dose est une sphère de 15 mm, 60 Gy : synthétique, car SynthRAD ne fournit pas de plans.");

  // ===== Section 3 : juges ======================================================================
  pres.addSection({ title: "Juges" });
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Juges" });
  s.addText("Comment on juge un juge", { placeholder: "title" });
  card(s, M, 1.5, 6.0, 4.85);
  text(s, "La consigne (identique pour les 10 VLM)", { x: M + 0.3, y: 1.7, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: C.accent1 });
  text(s, "« This image is an axial slice of a CT scan. It may have been generated by an AI model… Assess how anatomically " +
    "realistic it is: plausible anatomy, sharp boundaries, realistic contrast between bone, soft tissue and air, and absence " +
    "of artifacts (blur, noise, checkerboard patterns…). Give a score from 0 to 100 and a short justification. »",
    { x: M + 0.3, y: 2.2, w: 5.4, h: 2.0, fontSize: 15, italic: true, color: C.text2 });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 0.3, y: 4.45, w: 5.4, h: 0.6, rectRadius: 0.06,
    fill: { color: C.background1 }, line: { type: "none" }, objectName: id("json") });
  text(s, '{"score": 0–100, "reason": "…"}', { x: M + 0.45, y: 4.45, w: 5.1, h: 0.6, fontSize: 15, fontFace: "Courier New",
    valign: "middle", color: C.text2 });
  text(s, "Sortie JSON imposée · température 0 · graine 42 · 2 rendus (gray, rgb3) · 810 images par rendu",
    { x: M + 0.3, y: 5.25, w: 5.4, h: 0.9, fontSize: 14 });
  const crit = [
    ["chart", "ρ orienté", "Spearman entre note et MAE de la coupe : +1 parfait, 0 hasard, < 0 inversé"],
    ["layers", "ρ intra", "même patient, même coupe : seuls les 10 sCT changent"],
    ["sort", "ρ générateurs", "classement des 10 modèles de synthèse par note moyenne"],
    ["filter", "AUROC, filtrage", "séparer les coupes < 100 HU ; garder la moitié la mieux notée"],
  ];
  crit.forEach(([k, h, b], i) => {
    const y = 1.6 + i * 1.18;
    badge(s, 7.0, y, 0.62, k);
    text(s, h, { x: 7.8, y: y - 0.05, w: 4.9, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    text(s, b, { x: 7.8, y: y + 0.35, w: 4.9, h: 0.7, fontSize: 14 });
  });
  text(s, "Intervalles de confiance à 95 % : bootstrap sur les patients (1 000 tirages).",
    { x: 7.0, y: 6.3, w: 5.7, h: 0.4, fontSize: 11, color: C.accent6 });
  note(s, "La consigne a été écrite une fois et jamais réglée sur les résultats. Deux variantes existent dans le code " +
    "mais n'ont pas été évaluées. Le ρ intra est la mesure la plus stricte : il ne récompense pas un juge qui réagit " +
    "seulement à l'anatomie de la coupe.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Juges" });
  s.addText("Beaucoup de VLM ne discriminent pas", { placeholder: "title" });
  const iw = 7.7;
  s.addImage({ path: IMG("distributions_val_gray"), x: M, y: 1.55, w: iw, h: iw / 2.089, objectName: id("figure_distributions"),
    altText: "Histogrammes des notes de chaque VLM sur la validation, rendu gray." });
  text(s, "Distribution des notes, validation, rendu gray (810 images par VLM ; Qwen3-VL 8B : 638 ; MiniCPM-V 8B : 56).",
    { x: M, y: 1.55 + iw / 2.089 + 0.1, w: iw, h: 0.5, fontSize: 11, color: C.accent6 });
  const fails = [
    ["Note constante", "Moondream : 0 partout. MedGemma 27B : 75 pour 99,8 % des images."],
    ["Complaisance", "LLaVA 7B : 100 pour 75 % des images, en recopiant la liste d'artefacts de la consigne."],
    ["Boucles, coupures", "MiniCPM-V 8B : note valable pour 23 % des images. Qwen3-VL 8B : 17 % coupées (raisonnement trop long)."],
  ];
  fails.forEach(([h, b], i) => {
    const y = 1.55 + i * 1.72;
    card(s, 8.65, y, 4.08, 1.5);
    badge(s, 8.85, y + 0.2, 0.5, "warn", C.accent4);
    text(s, h, { x: 9.5, y: y + 0.2, w: 3.1, h: 0.5, fontSize: 17, bold: true, color: C.text2, valign: "middle" });
    text(s, b, { x: 8.85, y: y + 0.78, w: 3.75, h: 0.68, fontSize: 13 });
  });
  text(s, "Les bons juges utilisent 20 à 26 notes différentes ; MedGemma 4B n'en utilise que 4, mais dans le bon ordre.",
    { x: M, y: 6.05, w: iw, h: 0.6, fontSize: 14, bold: true, color: C.text2 });
  note(s, "Avant de corréler, on vérifie que les juges répondent et qu'ils discriminent. MedGemma 27B donne 75 à presque " +
    "tout, avec des justifications pourtant variées. Les erreurs de MiniCPM-V sont des boucles de génération arrêtées par " +
    "Ollama, des réponses coupées au plafond de 2 048 jetons ou des justifications vides. Le pipeline retente désormais ces images.");

  // Leaderboard (barres natives + barres d'erreur ajoutées après écriture)
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Juges" });
  s.addText("Qwen3-VL 30B juge le mieux, MedGemma 4B suit", { placeholder: "title" });
  const order = D.order.slice().reverse(); // PowerPoint place la première catégorie en bas
  const lbSeries = ["gray", "rgb3"].map((st) => ({ name: `rendu ${st}`, labels: order,
    values: order.map((m) => +D.val[st][m].rho.toFixed(3)) }));
  const errBars = ["gray", "rgb3"].map((st) => ({
    plus: order.map((m) => D.val[st][m].high - D.val[st][m].rho),
    minus: order.map((m) => D.val[st][m].rho - D.val[st][m].low) }));
  s.addChart(pres.charts.BAR, lbSeries, Object.assign({}, chartText, {
    x: M, y: 1.45, w: 7.9, h: 5.3, barDir: "bar", catAxisLabelPos: "low", barGrouping: "clustered", barGapWidthPct: 45,
    chartColors: [HEX.accent1, HEX.accent2], showLegend: true, legendPos: "b",
    valAxisMinVal: -0.4, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "ρ orienté avec la MAE de la coupe (IC 95 %)", valAxisTitleFontSize: 11,
    valAxisTitleColor: HEX.accent6, valAxisTitleFontFace: "+mn-lt", objectName: "graphique_leaderboard",
    altText: "Corrélation orientée de chaque VLM, rendus gray et rgb3, avec intervalles de confiance." }));
  const bv = D.val.gray[D.best_val], mg = D.val.gray["MedGemma 4B"];
  stat(s, 8.95, 1.5, 3.8, fr(bv.rho), `Qwen3-VL 30B, gray [${fr(bv.low)} ; ${fr(bv.high)}] : meilleur que chacun des 9 autres`);
  stat(s, 8.95, 3.15, 3.8, fr(mg.rho), `MedGemma 4B, gray [${fr(mg.low)} ; ${fr(mg.high)}] : deuxième, 4 milliards de paramètres`, C.accent3);
  stat(s, 8.95, 4.8, 3.8, "≤ 0,25", "pour les 8 autres VLM, quel que soit le rendu", C.accent6);
  note(s, "Validation, 810 images par rendu. Barres : intervalles de confiance à 95 % par bootstrap sur les patients. " +
    "La différence entre Qwen3-VL 30B et MedGemma 4B vaut 0,09 [0,02 ; 0,16] : significative. " +
    "Attention : même le meilleur juge n'explique qu'une petite part du classement des images.");

  // Effet du rendu
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Juges" });
  s.addText("Le rendu peut inverser un juge", { placeholder: "title" });
  const del = D.deltas.filter((d) => d.name !== "Moondream" && d.name !== "MiniCPM-V 8B").sort((a, b) => a.delta - b.delta);
  const cls = (d) => (d.low > 0 ? 0 : d.high < 0 ? 1 : 2);
  const delSeries = ["rgb3 meilleur", "gray meilleur", "pas de différence significative"].map((nm, k) => ({
    name: nm, labels: del.map((d) => d.name), values: del.map((d) => (cls(d) === k ? +d.delta.toFixed(3) : 0)) }));
  s.addChart(pres.charts.BAR, delSeries, Object.assign({}, chartText, {
    x: M, y: 1.45, w: 7.6, h: 5.3, barDir: "bar", catAxisLabelPos: "low", barGrouping: "clustered", barOverlapPct: 100, barGapWidthPct: 40,
    chartColors: [HEX.accent2, HEX.accent1, "AEB8BE"], showLegend: true, legendPos: "b",
    valAxisMinVal: -0.4, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
    showValAxisTitle: true, valAxisTitle: "Δρ = ρ rgb3 − ρ gray (mêmes coupes)", valAxisTitleFontSize: 11,
    valAxisTitleColor: HEX.accent6, valAxisTitleFontFace: "+mn-lt", objectName: "graphique_rendu",
    altText: "Différence de corrélation entre les rendus rgb3 et gray pour chaque VLM." }));
  const q38 = D.val, dMed = D.deltas.find((d) => d.name === "MedGemma 4B");
  stat(s, 8.7, 1.5, 4.0, `${fr(q38.gray["Qwen3.8 27B"].rho)} → +${fr(q38.rgb3["Qwen3.8 27B"].rho)}`,
    "Qwen3.8 27B passe d'un juge inversé à un juge utile", C.accent2);
  stat(s, 8.7, 3.15, 4.0, fr(dMed.delta), "MedGemma 4B perd tout son signal en rgb3, le codage conçu pour lui", C.accent1);
  card(s, 8.7, 4.95, 4.03, 1.7);
  badge(s, 8.9, 5.15, 0.5, "bulb", C.accent3);
  text(s, "Le rendu se choisit modèle par modèle, sur la validation.", { x: 9.55, y: 5.1, w: 3.05, h: 1.4, fontSize: 15, bold: true,
    color: C.text2, valign: "middle" });
  note(s, "gray : une fenêtre large de −1000 à 1000 HU. rgb3 : trois fenêtres dans les trois canaux, recommandées pour " +
    "MedGemma 1.5. Qwen3-VL 30B est insensible au rendu (−0,01). MiniCPM-V 8B (49 coupes appariées) et Moondream " +
    "(note constante) ne sont pas affichés.");

  // Test
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Juges" });
  s.addText("Le test confirme MedGemma 4B", { placeholder: "title" });
  const common = D.common.slice().sort((a, b) => D.val.gray[a].rho - D.val.gray[b].rho);
  s.addChart(pres.charts.BAR, [
    { name: "validation", labels: common, values: common.map((m) => +D.val.gray[m].rho.toFixed(3)) },
    { name: "test (patients jamais vus)", labels: common, values: common.map((m) => +D.test[m].rho.toFixed(3)) },
  ], Object.assign({}, chartText, {
    x: M, y: 1.45, w: 6.6, h: 4.0, barDir: "bar", catAxisLabelPos: "low", barGrouping: "clustered", barGapWidthPct: 50,
    chartColors: ["9DB7D8", HEX.accent1], showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", dataLabelColor: HEX.dk1, valAxisMinVal: -0.1, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0", objectName: "graphique_test",
    altText: "Corrélation orientée sur la validation et le test pour les quatre VLM évalués sur les deux." }));
  const mt = D.test["MedGemma 4B"];
  stat(s, M, 5.6, 2.0, fr(mt.rho), "ρ sur le test", C.accent1);
  stat(s, M + 2.2, 5.6, 2.0, fr(mt.gen), "ρ générateurs", C.accent1);
  stat(s, M + 4.4, 5.6, 2.2, fr(mt.auroc), "AUROC", C.accent1);
  card(s, 7.6, 1.5, 5.13, 3.3, "FBEFE0");
  badge(s, 7.85, 1.75, 0.62, "warn", C.accent2);
  text(s, "Limite : test incomplet", { x: 8.65, y: 1.75, w: 3.9, h: 0.62, fontSize: 19, bold: true, color: C.accent2, valign: "middle" });
  text(s, [
    { text: "Le benchmark de test a été arrêté après 4 modèles.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "Qwen3-VL 30B, meilleur juge sur la validation, n'a pas été évalué sur le test", options: { bold: true } },
    { text: ", ni Qwen3.8 27B, LLaVA 13B, MedGemma 27B et MiniCPM-V 8B.", options: { breakLine: true } },
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    { text: "À relancer en priorité : environ 4 h de calcul." },
  ], { x: 7.85, y: 2.5, w: 4.65, h: 2.2, fontSize: 15, color: C.text2 });
  card(s, 7.6, 5.05, 5.13, 1.6);
  badge(s, 7.85, 5.3, 0.5, "check", C.accent3);
  text(s, "Les 4 modèles évalués sur les deux parties gardent le même ordre : le classement se généralise à de nouveaux patients.",
    { x: 8.55, y: 5.2, w: 4.0, h: 1.3, fontSize: 14, valign: "middle" });
  note(s, "Les quatre modèles évalués sur les deux parties gardent le même ordre : le classement se généralise à des patients " +
    "nouveaux. Mais la première place de Qwen3-VL 30B ne repose que sur la validation, où il a aussi été sélectionné : " +
    "elle peut être un peu optimiste.");

  // ===== Section 4 : analyse ====================================================================
  pres.addSection({ title: "Analyse" });
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Analyse" });
  s.addText("Un juge de modèles plutôt qu'un juge d'images", { placeholder: "title" });
  const tiles = [
    [fr(bv.gen), "Qwen3-VL 30B : classement des 10 générateurs", C.accent1],
    [fr(bv.rho), "Qwen3-VL 30B : classement des 810 images", C.accent6],
    [fr(mt.gen), "MedGemma 4B (test) : classement des générateurs", C.accent3],
    [fr(mt.rho), "MedGemma 4B (test) : classement des images", C.accent6],
  ];
  tiles.forEach(([b, l, c], i) => stat(s, M + i * 3.08, 1.35, 2.85, b, l, c));
  const sw = 10.6;
  s.addImage({ path: IMG("scores_generateurs"), x: (W - sw) / 2, y: 3.05, w: sw, h: sw / 2.827, objectName: id("figure_generateurs"),
    altText: "Note moyenne de trois juges pour chaque générateur en fonction de sa MAE moyenne." });
  note(s, "En moyennant les notes sur les 81 images d'un générateur, le bruit d'un jugement individuel disparaît. " +
    "Ces juges sont donc utiles pour comparer des méthodes de synthèse, pas pour accepter un sCT. " +
    "Les pré-entraînés reçoivent des notes plus basses que le U-Net à MAE égale : les juges réagissent aussi au style d'image.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Analyse" });
  s.addText("Filtrer : utile pour écarter les pires sCT", { placeholder: "title" });
  const fv = D.filtering.val, xs = fv.fractions.map((f) => `${Math.round(f * 100)} %`);
  const fSeries = [["oracle (vraie MAE)", fv.oracle], ["hasard", fv.hasard], ["Qwen3-VL 30B", fv["Qwen3-VL 30B"]],
    ["MedGemma 4B", fv["MedGemma 4B"]], ["LLaVA 7B", fv["LLaVA 7B"]]].map(([nm, v]) => ({ name: nm, labels: xs,
    values: v.map((x) => +x.toFixed(1)) }));
  s.addChart(pres.charts.LINE, fSeries, Object.assign({}, chartText, {
    x: M, y: 1.45, w: 7.9, h: 5.3, chartColors: ["52514E", "9AA5AB", HEX.accent1, HEX.accent3, HEX.accent5],
    lineDataSymbol: "none", lineSize: 2.5, lineDash: "solid",
    showLegend: true, legendPos: "b", valAxisMinVal: 50, valAxisMaxVal: 140, valAxisMajorUnit: 10,
    showValAxisTitle: true, valAxisTitle: "MAE moyenne des coupes gardées (HU)", valAxisTitleFontSize: 11,
    valAxisTitleColor: HEX.accent6, valAxisTitleFontFace: "+mn-lt",
    showCatAxisTitle: true, catAxisTitle: "part des coupes gardées (les mieux notées d'abord)", catAxisTitleFontSize: 11,
    catAxisTitleColor: HEX.accent6, catAxisTitleFontFace: "+mn-lt", objectName: "graphique_filtrage",
    altText: "MAE moyenne des coupes gardées selon la part gardée, pour trois juges, l'oracle et le hasard (validation)." }));
  stat(s, 8.95, 1.5, 3.8, "136 → 114 HU", "MAE moyenne en gardant la moitié la mieux notée par Qwen3-VL 30B");
  stat(s, 8.95, 3.15, 3.8, `G = ${fr(bv.gain)}`, `du gain d'un oracle ; MedGemma 4B : ${fr(mg.gain)} (val), ${fr(mt.gain)} (test)`, C.accent3);
  card(s, 8.95, 4.95, 3.78, 1.7);
  text(s, "Étape suivante du vade-mecum, non faite : réentraîner un modèle sur les données filtrées.",
    { x: 9.15, y: 5.05, w: 3.4, h: 1.5, fontSize: 14, valign: "middle" });
  note(s, "Validation, rendu gray. G = 1 : aussi bien que l'oracle ; G = 0 : pas mieux que le hasard. " +
    "La courbe de MedGemma 4B est plate au début : avec quatre notes possibles, il ne trie pas finement le haut du classement.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Analyse" });
  s.addText("Les notes suivent l'apparence plus que la dose", { placeholder: "title" });
  const metLabels = ["MAE volume", "SSIM volume", "gamma photons", "gamma protons", "écart D98 protons"];
  s.addChart(pres.charts.BAR, Object.entries(D.dose.values).map(([nm, v]) => ({ name: nm, labels: metLabels,
    values: v.map((x) => +x.toFixed(3)) })), Object.assign({}, chartText, {
    x: M, y: 1.45, w: 7.9, h: 5.3, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [HEX.accent1, HEX.accent3], showLegend: true, legendPos: "b", showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", dataLabelColor: HEX.dk1, valAxisMinVal: 0, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.1,
    valAxisLabelFormatCode: "0.0", showValAxisTitle: true, valAxisTitle: "ρ orienté avec la note du VLM",
    valAxisTitleFontSize: 11, valAxisTitleColor: HEX.accent6, valAxisTitleFontFace: "+mn-lt", objectName: "graphique_dose",
    altText: "Corrélation entre la note du VLM et les métriques image et dose du volume (validation, gray)." }));
  card(s, 8.95, 1.5, 3.78, 2.45);
  badge(s, 9.15, 1.7, 0.5, "eye");
  text(s, "Le VLM voit une coupe 2D ; la dose dépend de la densité intégrée en 3D le long des faisceaux.",
    { x: 9.15, y: 2.35, w: 3.4, h: 1.5, fontSize: 15 });
  card(s, 8.95, 4.2, 3.78, 2.45, "DCE6F0");
  badge(s, 9.15, 4.4, 0.5, "balance", C.text2);
  text(s, "Un juge visuel ne remplace pas une vérification dosimétrique.", { x: 9.15, y: 5.05, w: 3.4, h: 1.4, fontSize: 17,
    bold: true, color: C.text2 });
  note(s, "Validation, rendu gray. Corrélations orientées : +1 veut dire que le VLM préfère les sCT qui ont la meilleure " +
    "métrique. Sur le test, MedGemma 4B donne 0,17 avec le gamma protons, comme sur la validation.");

  // Justifications
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Analyse" });
  s.addText("Ce que disent les justifications", { placeholder: "title" });
  const cols = ["Ép. 1 · 133 HU", "Ép. 100 · 49 HU", "Li · 63 HU"];
  const cx0 = 3.35, cwid = 1.15;
  cols.forEach((c, j) => text(s, c, { x: cx0 + j * cwid - 0.1, y: 1.45, w: cwid + 0.2, h: 0.5, fontSize: 12, bold: true,
    align: "center", color: C.accent6, valign: "bottom" }));
  text(s, "Extrait de justification", { x: 7.0, y: 1.45, w: 5.7, h: 0.5, fontSize: 12, bold: true, color: C.accent6, valign: "bottom" });
  const judges = [
    ["Qwen3-VL 30B", "décrit des défauts précis", [30, 95, 75],
      "« The image contains a prominent checkerboard pattern (a common AI-generated artifact)… » (ép. 1)"],
    ["MedGemma 4B", "même réponse aux trois", [85, 85, 85],
      "« The anatomy is generally plausible… There are no obvious artifacts. » (les trois sCT)"],
    ["LLaVA 7B", "recopie la consigne", [90, 100, 90],
      "« …no apparent artifacts such as blur, unusual noise, checkerboard patterns… » (liste de la consigne)"],
  ];
  const scoreColor = (v) => (v >= 90 ? HEX.accent3 : v >= 70 ? "8DBF9F" : v >= 50 ? "E9B872" : HEX.accent4);
  judges.forEach(([nm, tag, sc, q], i) => {
    const y = 2.1 + i * 1.5;
    card(s, M, y, CW, 1.3);
    text(s, [{ text: nm, options: { bold: true, fontSize: 17, color: C.text2, breakLine: true } },
      { text: tag, options: { fontSize: 13, color: C.accent6 } }], { x: M + 0.25, y: y + 0.15, w: 2.4, h: 1.0, valign: "middle" });
    sc.forEach((v, j) => {
      const d = 0.8, x = cx0 + j * cwid + (cwid - d) / 2;
      s.addShape(pres.shapes.OVAL, { x, y: y + 0.25, w: d, h: d, fill: { color: scoreColor(v) }, line: { type: "none" },
        objectName: id("note") });
      text(s, String(v), { x, y: y + 0.25, w: d, h: d, fontSize: 18, bold: true, align: "center", valign: "middle",
        color: v >= 90 || v < 50 ? C.background1 : C.text1 });
    });
    text(s, q, { x: 7.0, y: y + 0.12, w: 5.5, h: 1.06, fontSize: 13, italic: true, valign: "middle" });
  });
  text(s, "Même coupe (patient de validation 1BC027, 65 % de la hauteur, rendu gray) : exemple choisi pour illustrer, pas tiré au hasard.",
    { x: M, y: 6.6, w: CW, h: 0.35, fontSize: 11, color: C.accent6 });
  note(s, "Sur toute la validation, Qwen3-VL 30B affirme « pas d'artefact » pour 4 % des sCT de l'époque 1 et 59 % de l'époque 100 : " +
    "ses justifications suivent ses notes. Il signale un motif en damier sur 27 % des sCT de l'époque 1, un vrai défaut des " +
    "réseaux peu entraînés. LLaVA 7B affirme l'absence d'artefact dans 92 % des cas, y compris pour 93 % des sCT de l'époque 1.");

  // Coût
  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Analyse" });
  s.addText("Le coût : MedGemma 4B est huit fois plus léger", { placeholder: "title" });
  const cost = [
    ["MedGemma 4B", "modèle médical", C.accent3, [["0,5 s", "par image"], ["3,4 Go", "de VRAM"], ["7 min", "pour 810 images"], [fr(mg.rho), "ρ orienté (val)"]]],
    ["Qwen3-VL 30B", "généraliste, mélange d'experts", C.accent1, [["1,8 s", "par image"], ["26,9 Go", "de VRAM"], ["27 min", "pour 810 images"], [fr(bv.rho), "ρ orienté (val)"]]],
  ];
  cost.forEach(([nm, sub, col, items], i) => {
    const x = M + i * 4.35, w = 4.05;
    card(s, x, 1.5, w, 5.15);
    text(s, [{ text: nm, options: { bold: true, fontSize: 22, color: col, breakLine: true } }, { text: sub, options: { fontSize: 13, color: C.accent6 } }],
      { x: x + 0.3, y: 1.7, w: w - 0.6, h: 0.9 });
    items.forEach(([b, l], k) => {
      text(s, b, { x: x + 0.3, y: 2.75 + k * 0.95, w: 1.9, h: 0.6, fontSize: 28, bold: true, color: C.text2,
        fontFace: THEME.headFontFace, valign: "middle" });
      text(s, l, { x: x + 2.2, y: 2.75 + k * 0.95, w: 1.6, h: 0.6, fontSize: 14, valign: "middle" });
    });
  });
  card(s, 9.3, 1.5, 3.43, 5.15, "FBEFE0");
  badge(s, 9.5, 1.7, 0.5, "times", C.accent4);
  text(s, "Modèles denses de 27 B", { x: 10.15, y: 1.7, w: 2.45, h: 0.5, fontSize: 16, bold: true, color: C.text2, valign: "middle" });
  text(s, "2,7 à 5,6 s par image, plus d'une heure pour 810 images, pour un signal nul (MedGemma 27B) ou qui dépend entièrement du rendu (Qwen3.8 27B).",
    { x: 9.5, y: 2.4, w: 3.05, h: 2.7, fontSize: 14 });
  text(s, "Tous les VLM tiennent sur la RTX 5090 (32 Go), un à la fois.", { x: 9.5, y: 4.6, w: 3.05, h: 1.0, fontSize: 13, color: C.accent6 });
  note(s, "Latence médiane par image sur la validation, rendu gray. Qwen3-VL 30B reste rapide malgré sa taille grâce à son " +
    "architecture en mélange d'experts. En rgb3 (images de 896 pixels), la latence de MedGemma 4B passe de 0,5 à 3,1 s. " +
    "Référence du vade-mecum : SegAE évalue un masque 3D en 0,06 s.");

  // ===== Section 5 : conclusion =================================================================
  pres.addSection({ title: "Conclusion" });
  s = pres.addSlide({ masterName: "CONCLUSION_SOMBRE", sectionTitle: "Conclusion" });
  s.addText("Réponse : oui, partiellement", { placeholder: "title" });
  const concl = [
    ["check", C.accent3, "Ce qui marche", ["Classer des modèles de synthèse (ρ générateurs ≈ 0,85)", "Écarter grossièrement les pires sCT (gain de 0,2 à 0,4)"]],
    ["times", C.accent4, "Ce qui ne marche pas", ["Accepter ou refuser un sCT seul (AUROC 0,6 à 0,7)", "Remplacer une vérification de dose"]],
    ["bulb", C.accent2, "Recommandation", ["MedGemma 4B en gray pour un tri rapide et léger", "Qwen3-VL 30B si la VRAM le permet ; rendu et consigne réglés sur la validation"]],
  ];
  const keyStats = [["0,85", "ρ générateurs (Qwen3-VL 30B)"], ["0,6–0,7", "AUROC sur une image"], ["0,5 s", "par image avec MedGemma 4B"]];
  concl.forEach(([k, col, h, items], i) => {
    const x = M + i * (3.86 + 0.27);
    card(s, x, 1.6, 3.86, 4.6, C.background1);
    text(s, keyStats[i][0], { x: x + 0.3, y: 4.85, w: 3.3, h: 0.7, fontSize: 32, bold: true, color: col,
      fontFace: THEME.headFontFace, valign: "bottom" });
    text(s, keyStats[i][1], { x: x + 0.3, y: 5.6, w: 3.3, h: 0.4, fontSize: 13, color: C.accent6 });
    badge(s, x + 0.3, 1.85, 0.62, k, col);
    text(s, h, { x: x + 1.08, y: 1.85, w: 2.6, h: 0.62, fontSize: 19, bold: true, color: C.text2, valign: "middle" });
    text(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })),
      { x: x + 0.3, y: 2.75, w: 3.3, h: 2.0, fontSize: 15, paraSpaceAfter: 10 });
  });
  text(s, "Loin des 0,902 de SegAE, mais sans aucun entraînement : un outil d'aide, pas un dispositif de décision clinique.",
    { x: M, y: 6.35, w: CW, h: 0.5, fontSize: 15, italic: true, color: C.background2 });
  note(s, "Deux VLM sur dix donnent un signal significatif et stable. Le modèle médical de 4 milliards de paramètres fait " +
    "au moins aussi bien que tous les généralistes de 7 à 27 milliards, mais il est battu par le plus gros généraliste. " +
    "La version médicale de 27 milliards s'effondre : ni la spécialisation ni la taille ne garantissent un bon juge.");

  s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Conclusion" });
  s.addText("Limites et prochaines étapes", { placeholder: "title" });
  const steps = [
    ["Compléter le test", "6 modèles manquants, dont Qwen3-VL 30B, chacun dans son meilleur rendu (≈ 4 h)"],
    ["Tester d'autres consignes", "barème explicite, recherche de défauts, notation par paires"],
    ["Donner plus de contexte", "plusieurs coupes par sCT, ou l'IRM source à côté du sCT"],
    ["Combiner ou adapter un juge", "moyenne des rangs ; affinage léger de MedGemma 4B, dans l'esprit de SegAE"],
    ["Boucler le protocole", "réentraîner sur les données filtrées, puis passer au CBCT → CT"],
  ];
  steps.forEach(([h, b], i) => {
    const y = 1.5 + i * 1.04;
    numberBadge(s, M, y + 0.05, 0.55, i + 1);
    text(s, h, { x: M + 0.8, y, w: 6.7, h: 0.4, fontSize: 17, bold: true, color: C.text2 });
    text(s, b, { x: M + 0.8, y: y + 0.4, w: 6.7, h: 0.55, fontSize: 14 });
  });
  card(s, 8.45, 1.5, 4.28, 5.15, "FBEFE0");
  badge(s, 8.7, 1.72, 0.5, "warn", C.accent2);
  text(s, "Limites à garder en tête", { x: 9.35, y: 1.72, w: 3.2, h: 0.5, fontSize: 17, bold: true, color: C.accent2, valign: "middle" });
  const lim = ["Une seule consigne évaluée", "27 patients par partie : IC de ±0,07 à ±0,09", "Cible de dose synthétique (sphère, 60 Gy)",
    "Pas de sCT de qualité état de l'art", "20 configurations comparées sans correction", "Volet recalage hors de ce travail"];
  text(s, lim.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < lim.length - 1 } })),
    { x: 8.7, y: 2.45, w: 3.85, h: 4.0, fontSize: 14, paraSpaceAfter: 8 });
  note(s, "Compléter le test est la priorité : sans lui, la première place de Qwen3-VL 30B reste à confirmer. " +
    "Le rapport détaille chaque limite et la correspondance avec le vade-mecum, en annexe.");

  await pres.writeFile({ fileName: OUT });
  await addErrorBars(OUT, "graphique_leaderboard", errBars);
  await patchChart(OUT, "graphique_filtrage", dashSeries(["dash", "sysDot"]));
  await applyTheme(OUT, THEME);
  console.log("Écrit", OUT);
}

// Trouve le fichier XML du graphique nommé `objectName`.
async function chartPath(zip, objectName) {
  const slides = Object.keys(zip.files).filter((f) => /^ppt\/slides\/slide\d+\.xml$/.test(f));
  for (const sf of slides) {
    const xml = await zip.file(sf).async("string");
    if (!xml.includes(`name="${objectName}"`)) continue;
    const rels = await zip.file(sf.replace("slides/", "slides/_rels/") + ".rels").async("string");
    return `ppt/charts/${rels.match(/Target="[^"]*charts\/(chart\d+\.xml)"/)[1]}`;
  }
  throw new Error("graphique introuvable : " + objectName);
}

async function patchChart(file, objectName, transform) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const chartFile = await chartPath(zip, objectName);
  zip.file(chartFile, transform(await zip.file(chartFile).async("string")));
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

// Pointillés propres aux premières séries d'un graphique en lignes (pptxgenjs n'en met qu'un pour tout le graphique).
function dashSeries(styles) {
  return (xml) => {
    const parts = xml.split("<c:ser>");
    styles.forEach((st, i) => { parts[i + 1] = parts[i + 1].replace('<a:prstDash val="solid"/>', `<a:prstDash val="${st}"/>`); });
    return parts.join("<c:ser>");
  };
}

// Ajoute des barres d'erreur personnalisées (IC 95 %) aux séries d'un graphique en barres natif.
async function addErrorBars(file, objectName, bars) {
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  const slides = Object.keys(zip.files).filter((f) => /^ppt\/slides\/slide\d+\.xml$/.test(f));
  let chartFile = null;
  for (const sf of slides) {
    const xml = await zip.file(sf).async("string");
    if (!xml.includes(`name="${objectName}"`)) continue;
    const rels = await zip.file(sf.replace("slides/", "slides/_rels/") + ".rels").async("string");
    const m = rels.match(/Target="[^"]*charts\/(chart\d+\.xml)"/);
    chartFile = `ppt/charts/${m[1]}`;
  }
  if (!chartFile) throw new Error("graphique introuvable : " + objectName);
  let xml = await zip.file(chartFile).async("string");
  const lit = (vals) => `<c:numLit><c:formatCode>General</c:formatCode><c:ptCount val="${vals.length}"/>` +
    vals.map((v, i) => `<c:pt idx="${i}"><c:v>${Math.max(0, v).toFixed(4)}</c:v></c:pt>`).join("") + "</c:numLit>";
  let k = 0;
  xml = xml.replace(/<c:cat>/g, (match) => {
    const b = bars[k++];
    if (!b) return match;
    return `<c:errBars><c:errBarType val="both"/><c:errValType val="cust"/><c:noEndCap val="0"/>` +
      `<c:plus>${lit(b.plus)}</c:plus><c:minus>${lit(b.minus)}</c:minus>` +
      `<c:spPr><a:ln w="15875"><a:solidFill><a:srgbClr val="0E1B24"/></a:solidFill></a:ln></c:spPr></c:errBars>` + match;
  });
  if (k !== bars.length) throw new Error(`séries trouvées : ${k}, attendues : ${bars.length}`);
  zip.file(chartFile, xml);
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

main().catch((e) => { console.error(e); process.exit(1); });
