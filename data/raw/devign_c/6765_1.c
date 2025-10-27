void av_force_cpu_flags(int arg){

    if (   (arg & ( AV_CPU_FLAG_3DNOW    |

                    AV_CPU_FLAG_3DNOWEXT |

                    AV_CPU_FLAG_MMXEXT   |

                    AV_CPU_FLAG_SSE      |

                    AV_CPU_FLAG_SSE2     |

                    AV_CPU_FLAG_SSE2SLOW |

                    AV_CPU_FLAG_SSE3     |

                    AV_CPU_FLAG_SSE3SLOW |

                    AV_CPU_FLAG_SSSE3    |

                    AV_CPU_FLAG_SSE4     |

                    AV_CPU_FLAG_SSE42    |

                    AV_CPU_FLAG_AVX      |

                    AV_CPU_FLAG_AVXSLOW  |

                    AV_CPU_FLAG_XOP      |

                    AV_CPU_FLAG_FMA3     |

                    AV_CPU_FLAG_FMA4     |

                    AV_CPU_FLAG_AVX2     ))

        && !(arg & AV_CPU_FLAG_MMX)) {

        av_log(NULL, AV_LOG_WARNING, "MMX implied by specified flags\n");

        arg |= AV_CPU_FLAG_MMX;

    }



    cpu_flags = arg;

}
