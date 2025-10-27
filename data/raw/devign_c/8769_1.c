static int div_round (int dividend, int divisor)

{

    if (dividend > 0)

	return (dividend + (divisor>>1)) / divisor;

    else

	return -((-dividend + (divisor>>1)) / divisor);

}
