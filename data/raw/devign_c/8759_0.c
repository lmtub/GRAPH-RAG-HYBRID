static av_cold void uninit(AVFilterContext *ctx)

{

    EvalContext *eval = ctx->priv;

    int i;



    for (i = 0; i < 8; i++) {

        av_expr_free(eval->expr[i]);

        eval->expr[i] = NULL;

    }

}
