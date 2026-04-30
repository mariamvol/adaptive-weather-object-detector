import os
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch

from adaptive_infer import AdaptiveWeatherDetectorBundle


BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = Path(os.getenv("ADAPTIVE_BUNDLE_PATH", BASE_DIR / "adaptive_detector_bundle.pt"))
APP_TITLE = "Адаптивная система идентификации объектов"
AUTO_MODE = "Автоматически"


# Загрузка модели один раз при запуске приложения
if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Не найден файл модели: {BUNDLE_PATH}\n"
        "Положите adaptive_detector_bundle.pt в одну папку с app.py и adaptive_infer.py "
        "или задайте путь через переменную ADAPTIVE_BUNDLE_PATH."
    )

bundle = AdaptiveWeatherDetectorBundle(BUNDLE_PATH)
WEATHER_CLASSES = list(bundle.weather_classes)
ROUTE_MAP = dict(bundle.route_map)
DEVICE_NAME = "CUDA" if torch.cuda.is_available() else "CPU"


# Вспомогательные функции интерфейса
def _fmt_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _make_cards(predicted_weather, weather_confidence, expert_used, detections):
    count = len(detections)
    avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0.0

    return f"""
    <div class="cards-grid">
        <div class="metric-card">
            <div class="metric-label">Условие видимости</div>
            <div class="metric-value">{predicted_weather}</div>
            <div class="metric-sub">уверенность: {_fmt_percent(weather_confidence)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Выбранный эксперт</div>
            <div class="metric-value">{expert_used}</div>
            <div class="metric-sub">маршрут адаптивной системы</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Найдено объектов</div>
            <div class="metric-value">{count}</div>
            <div class="metric-sub">шт.</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Средняя уверенность</div>
            <div class="metric-value">{avg_conf:.3f}</div>
            <div class="metric-sub">по найденным объектам</div>
        </div>
    </div>
    """


def _make_log(predicted_weather, weather_confidence, expert_used, count, mode):
    mode_text = "автоматически" if mode == AUTO_MODE else "вручную"
    return f"""
### Лог принятия решения

➠ Изображение загружено  
➠ Условия видимости определены: **{predicted_weather}**  
➠ Способ определения условий: **{mode_text}**  
➠ Уверенность классификатора: **{weather_confidence:.3f}**  
➠ Выбран экспертный маршрут: **{expert_used}**  
➠ Детекция выполнена, найдено объектов: **{count}**
    """


def _detections_to_df(detections):
    rows = []
    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        rows.append({
            "ID": i,
            "Класс": det["class_name"],
            "Уверенность": round(det["confidence"], 3),
            "x1": round(x1, 1),
            "y1": round(y1, 1),
            "x2": round(x2, 1),
            "y2": round(y2, 1),
            "Площадь bbox": round(area, 1),
        })
    return pd.DataFrame(rows)


def _probs_to_df(weather_probs):
    rows = []
    for cls_name, prob in sorted(weather_probs.items(), key=lambda x: x[1], reverse=True):
        rows.append({
            "Условие": cls_name,
            "Вероятность": round(float(prob), 4),
            "Процент": _fmt_percent(float(prob)),
        })
    return pd.DataFrame(rows)


def _routes_to_html():
    items = "".join(
        f"<tr><td>{weather}</td><td>{expert}</td></tr>"
        for weather, expert in ROUTE_MAP.items()
    )
    return f"""
    <div class="about-card">
        <h3>Маршруты адаптивного выбора</h3>
        <table class="route-table">
            <thead><tr><th>Условие</th><th>Эксперт</th></tr></thead>
            <tbody>{items}</tbody>
        </table>
    </div>
    """


# Основная функция инференса
def analyze_image(image, mode, conf_threshold, imgsz):
    if image is None:
        empty_df = pd.DataFrame()
        return (
            None,
            "<div class='error-card'>Загрузите изображение для анализа.</div>",
            empty_df,
            empty_df,
            "### Лог принятия решения\nОжидание изображения.",
        )

    weather_override = None if mode == AUTO_MODE else mode

    try:
        result = bundle.predict(
            image=image,
            conf=float(conf_threshold),
            imgsz=int(imgsz),
            weather_override=weather_override,
        )

        raw_result = result["raw_result"]
        annotated = raw_result.plot()

        # Ultralytics чаще возвращает BGR, а Gradio ожидает RGB.
        if isinstance(annotated, np.ndarray) and annotated.ndim == 3:
            annotated = annotated[:, :, ::-1]

        detections = result["detections"]
        cards_html = _make_cards(
            predicted_weather=result["predicted_weather"],
            weather_confidence=result["weather_confidence"],
            expert_used=result["expert_used"],
            detections=detections,
        )
        log_md = _make_log(
            predicted_weather=result["predicted_weather"],
            weather_confidence=result["weather_confidence"],
            expert_used=result["expert_used"],
            count=len(detections),
            mode=mode,
        )

        return (
            annotated,
            cards_html,
            _detections_to_df(detections),
            _probs_to_df(result["weather_probs"]),
            log_md,
        )

    except Exception as exc:
        empty_df = pd.DataFrame()
        return (
            None,
            f"<div class='error-card'>Ошибка при анализе: {exc}</div>",
            empty_df,
            empty_df,
            f"### Лог принятия решения\n❌ Ошибка: `{exc}`",
        )


# CSS
CSS = """
:root {
    --dark: #07111f;
    --dark-2: #0d1b2e;
    --blue: #2563eb;
    --blue-2: #60a5fa;
    --muted: #64748b;
    --line: #e5e7eb;
    --card: rgba(255,255,255,0.92);
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #f8fafc 0%, #eef4ff 48%, #f8fafc 100%) !important;
    font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

#hero {
    background: linear-gradient(135deg, #07111f 0%, #0f2442 52%, #102a52 100%);
    color: white !important;
    padding: 28px 32px;
    border-radius: 24px;
    box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

#hero::after {
    content: "";
    position: absolute;
    right: -80px;
    top: -80px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(96,165,250,0.35), transparent 65%);
    pointer-events: none;
}

#hero h1 {
    margin: 0;
    font-size: 34px !important;
    font-weight: 850 !important;
    letter-spacing: -0.04em !important;
    color: #ffffff !important;
    position: relative;
    z-index: 1;
}

#hero p {
    margin: 8px 0 0 0;
    color: #e5edff !important;
    font-size: 16px !important;
    position: relative;
    z-index: 1;
}

#hero b,
#hero strong,
#hero span {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.panel {
    background: var(--card) !important;
    border: 1px solid rgba(148, 163, 184, 0.22) !important;
    border-radius: 22px !important;
    box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08) !important;
    padding: 18px !important;
}

.primary-button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    border-radius: 14px !important;
    color: white !important;
    border: none !important;
    min-height: 46px !important;
    font-weight: 700 !important;
}

.cards-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 4px 0 10px 0;
}

.metric-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    margin-bottom: 8px;
}

.metric-value {
    font-size: 30px;
    line-height: 1.05;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.04em;
}

.metric-sub {
    margin-top: 8px;
    color: #64748b;
    font-size: 13px;
}

.error-card {
    background: #fff1f2;
    color: #be123c;
    border: 1px solid #fecdd3;
    border-radius: 16px;
    padding: 16px;
    font-weight: 700;
}

.about-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0 12px 34px rgba(15, 23, 42, 0.07);
}

.about-card h3 {
    margin-top: 0;
}

.route-table {
    width: 100%;
    border-collapse: collapse;
    overflow: hidden;
    border-radius: 14px;
}

.route-table th, .route-table td {
    padding: 10px 12px;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
}

.route-table th {
    color: #334155;
    background: #f8fafc;
}

@media (max-width: 1000px) {
    .cards-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
@media (max-width: 640px) {
    .cards-grid {
        grid-template-columns: 1fr;
    }
}
"""


# Интерфейс Gradio Blocks
with gr.Blocks(
    title=APP_TITLE,
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        radius_size="lg",
    ),
) as demo:
    gr.HTML(
        f"""
        <div id="hero">
            <h1>{APP_TITLE}</h1>
            <p>Локальный демонстрационный стенд: анализ условий видимости → адаптивный выбор эксперта → детекция объектов.</p>
            <p>Модель: <b>{BUNDLE_PATH.name}</b> · устройство: <b>{DEVICE_NAME}</b></p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Анализ изображения"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, elem_classes="panel"):
                    gr.Markdown("### 1. Загрузка изображения")
                    image_input = gr.Image(
                        label="Загрузите изображение сцены",
                        type="pil",
                        height=330,
                    )

                    gr.Markdown("### 2. Параметры анализа")
                    mode_input = gr.Dropdown(
                        choices=[AUTO_MODE] + WEATHER_CLASSES,
                        value=AUTO_MODE,
                        label="Режим определения условий",
                    )
                    conf_input = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.25,
                        step=0.05,
                        label="Порог уверенности детекции",
                    )
                    imgsz_input = gr.Dropdown(
                        choices=[512, 640, 768, 896, 1024],
                        value=768,
                        label="Размер изображения для детектора",
                    )
                    analyze_button = gr.Button(
                        "▶ Запустить анализ",
                        elem_classes="primary-button",
                    )

                    gr.Markdown(
                        """
                        💡 В автоматическом режиме система сама определяет условие видимости и выбирает экспертный маршрут.
                        """
                    )

                with gr.Column(scale=8, elem_classes="panel"):
                    gr.Markdown("### 3. Результат анализа")
                    result_image = gr.Image(
                        label="Изображение с результатами детекции",
                        type="numpy",
                        height=540,
                    )
                    cards_output = gr.HTML()

                    with gr.Row():
                        with gr.Column(scale=6):
                            gr.Markdown("### 4. Найденные объекты")
                            detections_output = gr.Dataframe(
                                interactive=False,
                                wrap=True,
                                label="Таблица объектов",
                            )
                        with gr.Column(scale=4):
                            gr.Markdown("### 5. Вероятности условий")
                            probs_output = gr.Dataframe(
                                interactive=False,
                                wrap=True,
                                label="Классификация условий",
                            )

                    log_output = gr.Markdown("### Лог принятия решения\nОжидание запуска анализа.")

            analyze_button.click(
                fn=analyze_image,
                inputs=[image_input, mode_input, conf_input, imgsz_input],
                outputs=[result_image, cards_output, detections_output, probs_output, log_output],
            )

        with gr.Tab("О системе"):
            with gr.Row():
                with gr.Column(scale=6):
                    gr.HTML(
                        """
                        <div class="about-card">
                            <h3>Назначение приложения</h3>
                            <p>
                            Приложение демонстрирует работу адаптивной интеллектуальной системы идентификации объектов в различных условиях видимости.
                            Пользователь загружает изображение, после чего система определяет условия видимости,
                            выбирает подходящий экспертный маршрут и выполняет детекцию объектов.
                            </p>
                            <h3>Логика работы</h3>
                            <ol>
                                <li>Загрузка изображения пользователем.</li>
                                <li>Классификация условий видимости.</li>
                                <li>Выбор экспертной модели через адаптивный маршрутизатор.</li>
                                <li>Детекция объектов и визуализация результата.</li>
                                <li>Вывод таблицы найденных объектов и вероятностей условий.</li>
                            </ol>
                        </div>
                        """
                    )
                with gr.Column(scale=6):
                    gr.HTML(_routes_to_html())
                    gr.HTML(
                        f"""
                        <div class="about-card" style="margin-top: 14px;">
                            <h3>Техническая информация</h3>

                            <p><b>Устройство:</b> {DEVICE_NAME}</p>
                            <p><b>Классы условий:</b> {', '.join(WEATHER_CLASSES)}</p>
                        </div>
                        """
                    )


if __name__ == "__main__":
    #demo.queue().launch(inbrowser=True, show_error=True)
    demo.queue().launch(share=True, show_error=True)