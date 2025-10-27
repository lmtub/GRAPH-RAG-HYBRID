static void blend_image_rgb(AVFilterContext *ctx, AVFrame *dst, const AVFrame *src, int x, int y)

{

    blend_image_packed_rgb(ctx, dst, src, 0, x, y, 0);

}
