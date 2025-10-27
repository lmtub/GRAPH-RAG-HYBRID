void sws_rgb2rgb_init(int flags){

#if (defined(HAVE_MMX2) || defined(HAVE_3DNOW) || defined(HAVE_MMX))  && defined(CONFIG_GPL)

	if(flags & SWS_CPU_CAPS_MMX2)

		rgb2rgb_init_MMX2();

	else if(flags & SWS_CPU_CAPS_3DNOW)

		rgb2rgb_init_3DNOW();

	else if(flags & SWS_CPU_CAPS_MMX)

		rgb2rgb_init_MMX();

	else

#endif /* defined(HAVE_MMX2) || defined(HAVE_3DNOW) || defined(HAVE_MMX) */

		rgb2rgb_init_C();

}
