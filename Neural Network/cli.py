import argparse
from neural_style_transfer import load_image, im_convert, run_style_transfer

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer App")
    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--output', type=str, required=True, help='Output path for stylized image')
    parser.add_argument('--steps', type=int, default=300, help='Number of optimization steps')
    args = parser.parse_args()

    content = load_image(args.content)
    style = load_image(args.style)
    output = run_style_transfer(content, style, num_steps=args.steps)
    out_img = im_convert(output)
    out_img.save(args.output)
    out_img.show()

if __name__ == '__main__':
    main()
