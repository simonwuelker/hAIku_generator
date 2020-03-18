import java.awt.Color;
import java.awt.Graphics;
import java.awt.Point;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Snake {
	private List<Point> positions = new ArrayList<>();
	private Random rand = new Random();
	public Point head;
	public int dir = 3;
	public boolean crashOrEatMyself;
	public int count=0;
	public int color;

	
	
	/* weitere Attribute siehe UML-Diagramm */
	
	// Konstruktor
	public Snake(int x1, int y1, int x2, int y2, int x3, int y3,int color) {
		this.color = color;
		head = new Point(x1,y1);
		positions.add(new Point(head.x,head.y));
		positions.add(new Point(x2,y2));
		positions.add(new Point(x3,y3));
	}

	public void zeichne(Graphics g) {
		for (Point p : positions)
		{
			if(color == 0)g.setColor(Color.green);
			if(color == 1)g.setColor(Color.blue);
			g.fillRect(p.x * 16,p.y * 16,16,16 );
		}

	}
	
	public void tick()
	{
		// A.) letztes Element / Ende der Schlange löschen
		positions.remove(positions.size()-1);
		
		// B.) Head-Position anpassen
		if(dir == 1)head.x++;//Right
		if(dir == 2)head.x--;//Left
		if(dir == 3)head.y++;//Down
		if(dir == 4)head.y--;//Up
		
		// C.) Eatmyself OR Crash?
		if(positions.contains(head) || Game.grid[head.x][head.y]==1)crashOrEatMyself = true;
			
			
		// D.) Fruit?
		if(Game.grid[head.x][head.y]==2){
			positions.add(new Point(head.x,head.y));
			Game.grid[head.x][head.y] = 0;
			Game.grid[rand.nextInt(24)+1][rand.nextInt(24)+1]=2;
			count++;
			
		}
		
				
		// E.) Neue Head Position vorne einfügen
		positions.add(0,new Point(head.x,head.y)); 
	}
}