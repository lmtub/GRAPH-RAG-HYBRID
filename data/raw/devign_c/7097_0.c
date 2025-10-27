static inline void RET_CHG_FLOW (DisasContext *ctx)

{

    ctx->exception = EXCP_MTMSR;

}
