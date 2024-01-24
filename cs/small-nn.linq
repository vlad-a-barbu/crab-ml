<Query Kind="Program" />

void Main()
{
	var data = new Datum[] {
		new([1, 1], 1),
		new([1, 0], 0),
		new([0, 1], 0),
		new([0, 0], 0)
	};
	var hps = new HyperParams(sigmoid, 1e-2, 1e-2, 100_000);
	double[] ws = [Random.Shared.NextDouble(), Random.Shared.NextDouble()];
	double b = Random.Shared.NextDouble();
	
	for (var i = 0; i < hps.epochs; i++)
	{
		graddesc(hps, data, ws, ref b);
	}
	
	foreach (var datum in data)
	{
		var yh = forward(datum.xs, ws, b, hps.act);
		(yh, datum.y).Dump();
	}
}


record Datum(double[] xs, double y);
record HyperParams(Func<double, double> act, double eps, double lr, int epochs);

double sigmoid(double x)
	=> 1 / (1 + Math.Exp(-x));

double model(double[] xs, double[] ws, double b)
	=> xs.Zip(ws).Sum(x => x.First * x.Second) + b;

double forward(double[] xs, double[] ws, double b, Func<double, double> act)
	=> act(model(xs, ws, b));
	
double cost(HyperParams hps, Datum[] data, double[] ws, double b) 
{
	var result = 0.0;
	foreach (var datum in data)
	{
		var yh = forward(datum.xs, ws, b, hps.act);
		var dist = datum.y - yh;
		result += dist * dist;
	}
	return result;
}

void graddesc(HyperParams hps, Datum[] data, double[] ws, ref double b) 
{
	var c = cost(hps, data, ws, b);
	
	var gradws = new double[ws.Length];
	for (var i = 0; i < ws.Length; i++)
	{
		var wi = ws.ToArray();
		wi[i] += hps.eps;
		var dwi = (cost(hps, data, wi, b) - c) / hps.eps;
		gradws[i] = dwi * hps.lr;
	}
	
	var db = (cost(hps, data, ws, b + hps.eps) - c) / hps.eps;
	var gradb = db * hps.lr;

	for (var i = 0; i < ws.Length; i++)
	{
		ws[i] -= gradws[i];
	}
	b -= gradb;
}

