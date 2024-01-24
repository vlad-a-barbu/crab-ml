<Query Kind="Program" />

void Main()
{
	var n = 20;
	var data = Enumerable.Range(1, n).Select((x, _) => new Datum([x], x * 2)).ToArray();
	var hps = new HyperParams(identity, 1e-4, 1e-4, 1000);
	double[] ws = [Random.Shared.NextDouble()];
	double b = Random.Shared.NextDouble();
	
	var ntrain = n * 4 / 5;
	var train = data.Take(ntrain).ToArray();
	var test = data.Skip(ntrain).ToArray();
	
	for (var i = 0; i < hps.epochs; i++)
	{
		graddesc(hps, train, ws, ref b, debug: true);
	}
	
	foreach (var datum in test)
	{
		var yh = forward(datum.xs, ws, b, hps.act);
		(yh, datum.y).Dump();
	}
}

record Datum(double[] xs, double y);
record HyperParams(Func<double, double> act, double eps, double lr, int epochs);

double identity(double x)
	=> x;

double relu(double x)
	=> x > 0 ? x : 0;

double lrelu(double x)
	=> x > 0 ? x : 1e-2*x;

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

void graddesc(HyperParams hps, Datum[] data, double[] ws, ref double b, bool debug = false) 
{
	var c = cost(hps, data, ws, b);
	if (debug) 
		$"c = {c}; ws = [{string.Join(",", ws)}]; b = {b};".Dump();
	
	var gradws = new double[ws.Length];
	for (var i = 0; i < ws.Length; i++)
	{
		var wi = ws.ToArray();
		wi[i] += hps.eps;
		var dwi = (cost(hps, data, wi, b) - c) / hps.eps;
		gradws[i] = dwi;
	}
	var gradb = (cost(hps, data, ws, b + hps.eps) - c) / hps.eps;

	for (var i = 0; i < ws.Length; i++)
	{
		ws[i] -= gradws[i] * hps.lr;
	}
	b -= gradb * hps.lr;
}

