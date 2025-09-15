import base64
import sys

# コマンドライン引数から画像ファイル名を取得
if len(sys.argv) < 2:
    print("使い方: python encode_image.py <画像ファイル名>")
    sys.exit(1)

input_image_path = sys.argv[1]

# 画像を読み込んでBase64にエンコード
try:
    with open(input_image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # 結果をテキストファイルに保存
    with open("encoded_image.txt", "w") as text_file:
        text_file.write(encoded_string)
        
    print(f"'{input_image_path}' をエンコードし、'encoded_image.txt' に保存しました。")

except FileNotFoundError:
    print(f"エラー: ファイル '{input_image_path}' が見つかりません。")