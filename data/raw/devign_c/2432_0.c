static void blend_image_rgba_pm(AVFilterContext *ctx, AVFrame *dst, const AVFrame *src, int x, int y)

{

    blend_image_packed_rgb(ctx, dst, src, 1, x, y, 1);

}
