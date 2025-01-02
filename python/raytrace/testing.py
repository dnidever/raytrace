from raytrace import surface,optics,ray

def testlens():
    """ simple test of a lens and a light ray."""

    rr = ray.Ray([0.0,0.0,0.0],[1.0,0.0,0.0])
    ll = optics.Lens([0.0,0.0,0.0],[1.0,0.0,0.0])
