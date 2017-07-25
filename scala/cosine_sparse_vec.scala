def cosine(a: SparseVector, b: SparseVector): Double = {
	val intersection = a.indices.intersect(b.indices)
	val magnitudeA = intersection.map(x => Math.pow(a.apply(x), 2)).sum
	val magnitudeB = intersection.map(x => Math.pow(b.apply(x), 2)).sum
	intersection.map(x => a.apply(x) * b.apply(x)).sum / (Math.sqrt(magnitudeA) * Math.sqrt(magnitudeB))
}

def compute(v1: SparseVector, v2: SparseVector): Double = {
	val dotProduct = LinalgShim.dot(v1, v2)
	val norms = Vectors.norm(v1, 2) * Vectors.norm(v2, 2)
	1.0 - (math.abs(dotProduct) / norms)
}