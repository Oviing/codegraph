package app

type Shape interface {
	Area() int
}

type Rect struct {
	W int
}

func (r *Rect) Area() int {
	return r.W * 2
}

func helper() int { return 1 }

func Start() int {
	r := &Rect{}
	return r.Area() + helper()
}
