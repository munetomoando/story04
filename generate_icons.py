"""PWA用アイコンを生成するスクリプト（Pillow使用）"""
from PIL import Image, ImageDraw, ImageFont
import math

BRAND = (224, 122, 95)      # #E07A5F
BRAND_DARK = (200, 90, 66)  # #C85A42
WHITE = (255, 255, 255)
CREAM = (250, 248, 240)     # #FAF8F0

def draw_icon(size):
    """地図ピン＋フォークナイフのフラットアイコンを描画"""
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    s = size  # shorthand
    pad = int(s * 0.08)

    # 背景: 角丸正方形
    r = int(s * 0.22)  # 角丸の半径
    draw.rounded_rectangle(
        [pad, pad, s - pad, s - pad],
        radius=r,
        fill=BRAND,
    )

    # マップピン（中央やや上に配置）
    cx = s // 2
    pin_top = int(s * 0.18)
    pin_w = int(s * 0.32)  # ピンの幅
    pin_h = int(s * 0.38)  # ピンの楕円部分の高さ
    pin_bottom = pin_top + pin_h
    tail_len = int(s * 0.18)  # ピンの尖った部分

    # ピンの楕円（上部）
    draw.ellipse(
        [cx - pin_w // 2, pin_top, cx + pin_w // 2, pin_bottom],
        fill=WHITE,
    )

    # ピンの三角（下部の尖り）
    tri_w = int(pin_w * 0.45)
    draw.polygon(
        [
            (cx - tri_w, int(pin_bottom - pin_h * 0.25)),
            (cx + tri_w, int(pin_bottom - pin_h * 0.25)),
            (cx, pin_bottom + tail_len),
        ],
        fill=WHITE,
    )

    # ピンの中の丸（ドーナツの穴）
    inner_r = int(pin_w * 0.22)
    inner_cy = pin_top + pin_h // 2
    draw.ellipse(
        [cx - inner_r, inner_cy - inner_r, cx + inner_r, inner_cy + inner_r],
        fill=BRAND,
    )

    # フォーク＆ナイフ（ピンの下に大きめに配置）
    utensil_y = pin_bottom + tail_len + int(s * 0.02)
    utensil_h = int(s * 0.22)
    line_w = max(3, int(s * 0.035))

    # フォーク（左）
    fork_x = cx - int(s * 0.12)
    # 持ち手
    draw.line([(fork_x, utensil_y), (fork_x, utensil_y + utensil_h)], fill=WHITE, width=line_w)
    # 歯（3本）
    tine_w = max(2, int(line_w * 0.7))
    tine_gap = int(s * 0.025)
    for dx in [-tine_gap, 0, tine_gap]:
        draw.line(
            [(fork_x + dx, utensil_y - int(utensil_h * 0.35)), (fork_x + dx, utensil_y + int(utensil_h * 0.1))],
            fill=WHITE, width=tine_w
        )

    # ナイフ（右）
    knife_x = cx + int(s * 0.12)
    draw.line([(knife_x, utensil_y - int(utensil_h * 0.35)), (knife_x, utensil_y + utensil_h)], fill=WHITE, width=line_w)
    # 刃の部分（太め）
    blade_w = max(3, int(s * 0.055))
    draw.line(
        [(knife_x, utensil_y - int(utensil_h * 0.35)), (knife_x, utensil_y + int(utensil_h * 0.2))],
        fill=WHITE, width=blade_w
    )

    return img


def main():
    import os
    out_dir = os.path.join(os.path.dirname(__file__), 'static', 'images')
    os.makedirs(out_dir, exist_ok=True)

    for sz in [192, 512]:
        icon = draw_icon(sz)
        path = os.path.join(out_dir, f'icon-{sz}.png')
        icon.save(path, 'PNG')
        print(f'Generated: {path}')

    # Apple Touch Icon (180x180)
    apple_icon = draw_icon(180)
    apple_path = os.path.join(out_dir, 'apple-touch-icon.png')
    apple_icon.save(apple_path, 'PNG')
    print(f'Generated: {apple_path}')


if __name__ == '__main__':
    main()
