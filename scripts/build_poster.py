"""Build the research poster directly from the scored model artifact."""
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
NAVY = colors.HexColor("#0b1529")
INK = colors.HexColor("#17263b")
MUTED = colors.HexColor("#506078")
ORANGE = colors.HexColor("#d75a26")
BLUE = colors.HexColor("#235b9d")
PAPER = colors.HexColor("#f3f5f9")


def build():
    artifact = json.loads((ROOT/"data/league-model.json").read_text())
    target = ROOT/"research/nba-matchup-poster.pdf"
    width, height = landscape(A3)
    canvas = Canvas(str(target), pagesize=(width,height))
    canvas.setTitle("NBA Matchup Lab — Predicting winners and point margins")
    canvas.setAuthor("Myke Tzul")
    canvas.setFillColor(PAPER); canvas.rect(0,0,width,height,fill=1,stroke=0)
    canvas.setFillColor(NAVY); canvas.rect(0,height-154,width,154,fill=1,stroke=0)
    canvas.setFillColor(colors.HexColor("#ffab78"))
    canvas.setFont("Helvetica-Bold",11); canvas.drawString(36,height-31,"NBA MATCHUP LAB  /  LEAGUE-WIDE FORECASTING")
    canvas.setFillColor(colors.white); canvas.setFont("Helvetica-Bold",32)
    canvas.drawString(36,height-75,"Can team form predict the next NBA game?")
    canvas.setFont("Helvetica",15)
    canvas.drawString(36,height-108,"Winner probabilities and point-margin forecasts across all 30 teams")
    canvas.setFont("Helvetica",11)
    canvas.drawString(36,height-133,"Myke Tzul  |  Research poster  |  November 2026")
    gap=18; col=(width-72-gap*2)/3
    left,middle,right=36,36+col+gap,36+2*(col+gap)
    top=height-178
    style=ParagraphStyle("body",fontName="Helvetica",fontSize=11.6,leading=16,textColor=INK,alignment=TA_LEFT)
    small=ParagraphStyle("small",parent=style,fontSize=10.3,leading=14,textColor=MUTED)
    def text(value,x,y,style_=style,w=col):
        para=Paragraph(value,style_); _,h=para.wrap(w,1000); para.drawOn(canvas,x,y-h);return y-h-12
    def title(value,x,y):
        canvas.setFillColor(ORANGE); canvas.setFont("Helvetica-Bold",14);canvas.drawString(x,y-14,value);return y-28
    y=title("QUESTION & DATA",left,top)
    y=text("Can earlier team strength, form, and rest improve on simple home-court and Elo baselines?",left,y)
    y=text("<b>4,920 regular-season games</b><br/>2022–23 through 2025–26<br/>One row per game. Final test: 1,230 games.",left,y)
    y=title("CHRONOLOGICAL DESIGN",left,y-2)
    for label,body in [
        ("FIT · 2022–23 & 2023–24","Fit the candidate models on earlier seasons."),
        ("SELECT · 2024–25","Choose win model by Brier error; margin model by MAE."),
        ("FINAL TEST · 2025–26","Refit through 2024–25, then score all candidates. Keep the prior selection."),
    ]:
        canvas.setFillColor(colors.white);canvas.roundRect(left,y-64,col,64,7,fill=1,stroke=0)
        canvas.setFillColor(BLUE);canvas.setFont("Helvetica-Bold",11);canvas.drawString(left+12,y-17,label)
        text(body,left+12,y-25,small,col-24);y-=76
    y=title("PREGAME INPUTS",left,y-3)
    y=text("Home court · Elo gap · smoothed last-10 scoring margins · smoothed last-10 win rates · capped rest-day gap.",left,y)
    text("All games on a date receive features before any result from that date updates team form. Refitted deployment models use all four historical seasons.",left,y,small)
    y=title("FINAL-TEST RESULTS",middle,top)
    selected_w=artifact["selected_win"];selected_m=artifact["selected_margin"]
    win=artifact["final_test"]["wins"][selected_w];margin=artifact["final_test"]["margins"][selected_m]
    y=text(f"<b>{selected_w}</b>",middle,y)
    metric_style=ParagraphStyle("metric",parent=style,fontSize=24,leading=30,textColor=BLUE)
    y=text(f"{win['accuracy']:.1%} winner accuracy",middle,y,metric_style)
    y=text(f"Brier error {win['brier']:.4f}; home baseline 0.2470.",middle,y,small)
    y=text(f"<b>{selected_m}</b>",middle,y)
    y=text(f"{margin['mae']:.2f} points MAE",middle,y,metric_style)
    y=text("Constant home-margin baseline: 13.10 points.",middle,y,small)
    y=title("MARGIN ERROR COMPARISON",middle,y-4)
    for name,row in artifact["final_test"]["margins"].items():
        canvas.setFillColor(INK);canvas.setFont("Helvetica",10);canvas.drawString(middle,y-12,name)
        canvas.setFillColor(ORANGE if name==selected_m else BLUE)
        canvas.rect(middle,y-29,col*.8*row["mae"]/14,10,fill=1,stroke=0)
        canvas.setFont("Helvetica-Bold",10);canvas.drawRightString(middle+col,y-29,f"{row['mae']:.2f}")
        y-=47
    y=text("MAE in points. Lower is better. All models use the same final-test games.",middle,y,small)
    y=title("DOES A BLEND HELP?",middle,y)
    text("A fixed 50/50 probability blend scored Brier 0.20645 versus 0.20650 for logistic regression on the final test. Logistic regression won on development. Ridge also beat the margin blend.",middle,y)
    y=title("WHAT THE NUMBERS MEAN",right,top)
    y=text("The models improve on constant home-court baselines, but single-game forecasts remain uncertain. A predicted five-point edge can easily be wrong.",right,y)
    radius=artifact["margin_band_radius"]
    y=text(f"An empirical range of <b>forecast ± {radius:.1f} points</b>, set from development residuals, covered <b>{artifact['final_band_coverage']:.1%}</b> of final-test margins against an 80% target.",right,y)
    y=title("UNCERTAINTY CHECK",right,y)
    band=artifact["paired_bootstrap"]["mae"]
    y=text(f"MAE difference versus baseline: {band['difference']:+.2f} points.<br/>95% date-block bootstrap interval: [{band['low']:+.2f}, {band['high']:+.2f}].<br/>500 resamples, seed 42. Cross-date dependence remains.",right,y)
    y=title("LIMITATIONS",right,y)
    y=text("• No injury, lineup, player-availability, travel, or odds inputs.<br/>• Two later seasons; no external prospective validation.<br/>• Historical data was already available in this repo.<br/>• Team statistics omit individual matchups.<br/>• Hypothetical future games hold current known form fixed.",right,y)
    y=title("LIVE PRODUCT",right,y)
    text("All-team matchup explorer, immutable pregame forecast records, and a separate Knicks Corner. New league picks are graded only after their recorded tipoff.",right,y)
    canvas.setFillColor(NAVY);canvas.rect(0,0,width,69,fill=1,stroke=0)
    canvas.setFillColor(colors.white);canvas.setFont("Helvetica",9)
    canvas.drawString(36,47,"Data: nba_data archive → NBA Cloud Data Platform historical game aggregation. Current schedule and scores: ESPN.")
    canvas.drawString(36,32,"Methods: scikit-learn logistic regression, ridge regression, gradient boosting, Elo, and fixed blends. Source and prediction CSV are in the repo.")
    link="https://github.com/Mykeil-tzul/knicks-win-probability-model"
    canvas.drawString(36,17,link+"   |   History SHA-256 "+artifact["history_sha256"][:16])
    canvas.linkURL(link,(36,10,690,30),relative=0)
    canvas.save()
    print(target)


if __name__=="__main__":
    build()
