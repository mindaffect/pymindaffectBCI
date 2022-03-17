package nl.ma.utopiaserver;
public interface ClockAlignment {
    public void addPoint(double X, double Y);
    public double getX(double Y);
    public double getY(double X);
    public double getYErr();
    public void reset();
}
