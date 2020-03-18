import java.awt.Canvas;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.image.BufferStrategy;
import java.util.Random;
import javax.swing.JCheckBoxMenuItem;
import javax.swing.JDialog;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPopupMenu;
import javax.swing.JRadioButtonMenuItem;
import javax.swing.JSeparator;
import javax.swing.border.Border;
import javax.swing.border.LineBorder;

import javax.swing.JFrame;
import javax.swing.JLabel;

// Runnable wegen Threads, Canvas für Zeichenfläche, KeyListener für Tastatureingaben
public class Game extends Canvas implements Runnable, KeyListener {
	// Globale Variable
	private Thread thread;
	private boolean isRunning = false;
	
	public static final int WIDTH = 416;
	public static final int HEIGHT = 416;

	public static final int COLS = 26;
	public static final int ROWS = 26;
	public static final int EMPTY = 0;
	public static final int WALL = 1;
	public static final int FRUIT = 2;
	private Random rand = new Random();
	Snake snake;
	Snake s;

	// Das Spielfeld
	public static int[][] grid = new int[ROWS][COLS];
	int[][] gridOrig = new int[][] { // 0...25
			{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
			{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };


	public synchronized void start() {
		if (isRunning)
			return;
		isRunning = true;
		thread = new Thread(this);
		thread.start();
	}

	public synchronized void stop() {
		if (!isRunning)
			return;
		isRunning = false;
		try {
			thread.join();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public Game() {
		Dimension dimension = new Dimension(WIDTH, HEIGHT);
		setPreferredSize(dimension);
		setMinimumSize(dimension);
		setMaximumSize(dimension);

		init();

		addKeyListener(this);
	}

	
	
	// I. Starteinstellungen / auch für Restart: init()
	private void init() {

		// Grid transponieren d.h. Spalten und Zeilen vertauschen
		for (int i = 0; i < COLS; i++)
			for (int j = 0; j < ROWS; j++)
				grid[i][j] = gridOrig[j][i];

		// SNAKE-Objekt erzeugen
		snake = new Snake(4,6,4,5,4,4,0);
		s = new Snake(7,6,7,5,7,4,1);
		Game.grid[rand.nextInt(24)+1][rand.nextInt(24)+1]=2;
	}

	
	
	// II. Positions-Updates: tick()
	private void tick() {
		// A.  Positions Update Snake
		snake.tick();
		s.tick();
		if (snake.crashOrEatMyself)init();
		if (s.crashOrEatMyself)init();

	}



	// III. Alles Zeichnen: render()
	// --------------------------------------------------------
	private void render() {
		BufferStrategy bs = getBufferStrategy();
		if (bs == null) {
			createBufferStrategy(3); // triple-Buffering // 2 geht auch
			return;
		}

		Graphics g = bs.getDrawGraphics();

		g.setColor(Color.white); // weissen Hintergrund malen
		g.fillRect(0, 0, WIDTH, HEIGHT);


		for (int i = 0; i < COLS; i++)
			for (int j = 0; j < ROWS; j++)
				if (grid[i][j] == WALL) {
					g.setColor(Color.black);
					g.fillRect(i * 16, j * 16, 16, 16);
				} else if (grid[i][j] == FRUIT) {
					g.setColor(Color.orange);
					g.fillRect(i * 16, j * 16, 16, 16);
				}

		/* SNAKE zeichnen */
		snake.zeichne(g);
		s.zeichne(g);
		g.setColor(Color.blue);
		g.drawString(String.valueOf(snake.count), 16*24, 30);
		g.dispose();
		bs.show();
	}

	
	
	// IV. Game-Loop
	@Override
	public void run() {
		try {
			Thread.sleep(200);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		// damit man nicht erst auf Programm klicken muss, damit
		// es auf Keyboard-Eingaben reagiert
		requestFocus();

		// mit 60 frames pro sekunde laufen lassen
		int fps = 0;
		double timer = System.currentTimeMillis();
		long lastTime = System.nanoTime();
		double targetTick = 45;
		double delta = 0;
		double ns = 1000000000 / targetTick;

		// Game-Loop
		while (isRunning) {
			long now = System.nanoTime();
			delta += (now - lastTime) / ns;
			lastTime = now;

			while (delta >= 1) {
				if (fps % 5 == 0)
				tick(); // Geschw. der Figuren etwas ausbremsen
				render();
				fps++;
				delta--;
			}

			// Zur Kontrolle jede Sekunde die Frames pro Sekunde ausgeben
			if (System.currentTimeMillis() - timer > 1000) {
				System.out.println(fps);
				fps = 0;
				timer += 1000;
			}

		}

		stop();

	}

	

	// V. Steuerung des PACMANs über die Pfeiltasten: keyPressed, keyReleased
	@Override
	public void keyPressed(KeyEvent e) {
		// Pfeiltasten gedrückt
		if (e.getKeyCode() == KeyEvent.VK_RIGHT&&snake.dir!=2) snake.dir = 1;
		if (e.getKeyCode() == KeyEvent.VK_LEFT&&snake.dir!=1) snake.dir = 2;
		if (e.getKeyCode() == KeyEvent.VK_UP&&snake.dir!=3) snake.dir = 4;
		if (e.getKeyCode() == KeyEvent.VK_DOWN&&snake.dir!=4) snake.dir = 3;
		if (e.getKeyCode() == KeyEvent.VK_D&&s.dir!=2) s.dir = 1;
		if (e.getKeyCode() == KeyEvent.VK_A&&s.dir!=1) s.dir = 2;
		if (e.getKeyCode() == KeyEvent.VK_W&&s.dir!=3) s.dir = 4;
		if (e.getKeyCode() == KeyEvent.VK_S&&s.dir!=4) s.dir = 3;
		
		
	

	}

	@Override
	public void keyReleased(KeyEvent e) {
		// Pfeiltasten losgelassen
		// * ... */
	}

	
	// Unnötig (darf man trotzdem nicht löschen)
	@Override
	public void keyTyped(KeyEvent e) {
		// TODO Auto-generated method stub
		
	}
	
	
	public static void main(String[] args) {
		Game game = new Game();
		JFrame frame = new JFrame();
		/* Erzeugung eines neuen Dialoges */
      /*  JDialog meinJDialog = new JDialog();
        meinJDialog.setTitle("JMenuBar für unser Java Tutorial Beispiel.");
        // Wir setzen die Breite auf 450 und die Höhe auf 300 Pixel
        meinJDialog.setSize(450,300);
        // Zur Veranschaulichung erstellen wir hier eine Border
        Border bo = new LineBorder(Color.yellow);
        // Erstellung einer Menüleiste
        JMenuBar bar = new JMenuBar();
        // Wir setzen unsere Umrandung für unsere JMenuBar
        bar.setBorder(bo);
        // Erzeugung eines Objektes der Klasse JMenu
        JMenu menu = new JMenu("Ich bin ein JMenu");
        // Erzeugung eines Objektes der Klasse JMenuItem
        JMenuItem item = new JMenuItem("BOOOM");
        JMenuItem item2 = new JMenuItem("JMenuItem innerhalb eines JPopups");
        // Wir fügen das JMenuItem unserem JMenu hinzu
        menu.add(item);
        // Erzeugung eines Objektes der Klasse JSeparator
        JSeparator sep = new JSeparator();
        JSeparator sep2 = new JSeparator();
        // JSeparator wird unserem JMenu hinzugefügt         
        menu.add(sep);
        // Erzeugung eines Objektes der Klasse JCheckBoxMenuItem
        JCheckBoxMenuItem checkBoxItem = new JCheckBoxMenuItem
            ("Ich bin das JCheckBoxMenuItem");
        JCheckBoxMenuItem checkBoxItem2 = new JCheckBoxMenuItem
            ("JCheckBoxMenuItem innerhalb eines JPopups");
        // JCheckBoxMenuItem wird unserem JMenu hinzugefügt         
        menu.add(checkBoxItem);
        // Erzeugung eines Objektes der Klasse JRadioButtonMenuItem
        JRadioButtonMenuItem radioButtonItem = new JRadioButtonMenuItem
            ("Ich bin ein JRadionButtonMenuItem",true);
        JRadioButtonMenuItem radioButtonItem2 = new JRadioButtonMenuItem
            ("JRadionButtonMenuItem innerhalb eines JPopups",true);
        // JRadioButtonMenuItem wird unserem JMenu hinzugefügt           
        menu.add(radioButtonItem);
        // Menü wird der Menüleiste hinzugefügt         
        bar.add(menu);
        // Erzeugung eines JPopupMenu-Objektes
        JPopupMenu pop = new JPopupMenu();
        // Wir setzen die Position unseres Kontextmenüs 
        // auf die Koordinaten X = 100 und Y =100
        pop.setLocation(100, 100);
        // Wir fügen unsere Menüeinträge unserem Kontextmenü hinzu
        pop.add(item2);
        pop.add(sep2);
        pop.add(checkBoxItem2);
        pop.add(radioButtonItem2);
        // Menüleiste wird für den Dialog gesetzt
        meinJDialog.setJMenuBar(bar);
        // Wir lassen unseren Dialog anzeigen
        meinJDialog.setVisible(true);
        // Wir lassen unser JPopupMenu anzeigen*/
       // pop.setVisible(true);
		frame.add(game);
		frame.setResizable(false);
		frame.pack();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setLocationRelativeTo(null); // Frame zentrieren
		frame.setVisible(true);
		game.start();
	}

}
