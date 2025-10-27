void checkasm_check_blend(void)

{

    uint8_t *top1 = av_malloc(BUF_SIZE);

    uint8_t *top2 = av_malloc(BUF_SIZE);

    uint8_t *bot1 = av_malloc(BUF_SIZE);

    uint8_t *bot2 = av_malloc(BUF_SIZE);

    uint8_t *dst1 = av_malloc(BUF_SIZE);

    uint8_t *dst2 = av_malloc(BUF_SIZE);

    FilterParams param = {

        .opacity = 1.0,

    };



#define check_and_report(name, val)               \

    param.mode = val;                             \

    ff_blend_init(&param, 0);                     \

    if (check_func(param.blend, #name))           \

        check_blend_func();



    check_and_report(addition, BLEND_ADDITION)

    check_and_report(addition128, BLEND_ADDITION128)

    check_and_report(and, BLEND_AND)

    check_and_report(average, BLEND_AVERAGE)

    check_and_report(darken, BLEND_DARKEN)

    check_and_report(difference128, BLEND_DIFFERENCE128)

    check_and_report(hardmix, BLEND_HARDMIX)

    check_and_report(lighten, BLEND_LIGHTEN)

    check_and_report(multiply, BLEND_MULTIPLY)

    check_and_report(or, BLEND_OR)

    check_and_report(phoenix, BLEND_PHOENIX)

    check_and_report(screen, BLEND_SCREEN)

    check_and_report(subtract, BLEND_SUBTRACT)

    check_and_report(xor, BLEND_XOR)

    check_and_report(difference, BLEND_DIFFERENCE)

    check_and_report(extremity, BLEND_EXTREMITY)

    check_and_report(negation, BLEND_NEGATION)



    report("8bit");



    av_freep(&top1);

    av_freep(&top2);

    av_freep(&bot1);

    av_freep(&bot2);

    av_freep(&dst1);

    av_freep(&dst2);

}
