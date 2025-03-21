from copy import copy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class Play:
    """
    Interactive game visualization class that handles:
    - Game display using matplotlib
    - Human-AI and AI-AI interactions
    - Move visualization
    - Game state management
    """
    
    def __init__(self, game, player1=None, player2=None, name='TicTacToe', mode_name=None):
        """
        Initialize the game visualization.
        """
        plt.close('all')  # Close any existing matplotlib windows
        
        self.original_game = game  # Store original game state for resets
        self.game = copy(game)     # Working copy of the game
        self.player1 = player1     # First player (AI or None)
        self.player2 = player2     # Second player (AI or None)
        self.player = self.game.player  # Current player
        self.end = False           # Game end flag
        self.fig = None            # Matplotlib figure
        self.ax = None            # Matplotlib axes
        self.click_cid = None     # Click event connection ID
        self.mode_name = mode_name # Game mode name
        self.anim = None          # Animation object
        self.setup_figure(name)   # Initialize the display
        self.reset()              # Reset game state
        plt.show()                # Display the game window

    def setup_figure(self, name='TicTacToe'):
        """
        Setup the matplotlib figure and controls.
        
        Args:
            name: Window title for the game (default: 'TicTacToe')
        """
        # Adjust figure size based on board dimensions
        if self.game.w * self.game.h < 25:
            figsize = (self.game.w, self.game.h)
        else:
            figsize = (self.game.w / 1.2, self.game.h / 1.2)

        # Create and configure the figure with new title
        self.fig = plt.figure(name, figsize=figsize)
        self.fig.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)
            
        # Create main game board axes
        self.ax = self.fig.add_axes([0.2, 0.2, 0.6, 0.6])
        
        # Add title showing game mode
        if self.mode_name:
            self.title = self.fig.suptitle(self.mode_name, fontsize=10, y=0.95)

        # Add restart button
        button_ax = self.fig.add_axes([0.4, 0.05, 0.2, 0.08])
        self.restart_button = Button(button_ax, 'Restart', 
                                color='black',
                                hovercolor='gray')
        self.restart_button.label.set_color('white')
        self.restart_button.ax.set_facecolor('black')
        self.restart_button.on_clicked(lambda event: self.reset())

    def reset(self):
        """Reset the game to initial state."""
        # Stop any existing animation
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                if hasattr(self.anim, 'event_source') and self.anim.event_source is not None:
                    self.anim.event_source.stop()
            except Exception as e:
                print(f"Warning: Could not stop animation: {e}")
            finally:
                self.anim = None
                    
        self.game = copy(self.original_game)
        if self.click_cid and self.fig:
            self.fig.canvas.mpl_disconnect(self.click_cid)
        self.click_cid = None
        self.end = False
        
        # Reset the title to original mode name
        if self.mode_name and hasattr(self, 'title'):
            self.title.set_text(self.mode_name)
        
        self.ax.clear()
        self.setup_grid()
        
        # Handle AI-AI game
        if self.player1 is not None and self.player2 is not None:
            self.setup_animation()
        else:
            # Handle games with at least one human player
            # If player1 is AI, make first move
            if self.player1 is not None:
                succeed = False
                while not succeed:
                    loc = self.player1(self.game)
                    succeed = self.game.move(loc)
                self.draw_move(loc)
            
            # Setup click handler for human moves
            if self.click_cid:
                self.fig.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = self.fig.canvas.mpl_connect('button_press_event', self.click)
        
        self.fig.canvas.draw_idle()

    def setup_grid(self):
        """
        Setup the game board grid.
        
        Creates:
        - Grid lines
        - Board boundaries
        - Proper scaling and aspect ratio
        """
        w, h = self.game.size
        
        # Set the exact limits without any margin
        self.ax.set_xlim([0, w])
        self.ax.set_ylim([0, h])
        
        # Draw the grid manually for consistent line width
        # Vertical lines
        for i in range(w+1):
            self.ax.axvline(x=i, color='black', linewidth=2.0)
        # Horizontal lines
        for i in range(h+1):
            self.ax.axhline(y=i, color='black', linewidth=2.0)
        
        # Remove ticks completely
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Configure display properties
        self.ax.format_coord = lambda x, y: ''
        self.ax.set_aspect('equal')

        # Remove border spines
        for loc in ['top', 'right', 'bottom', 'left']:
            self.ax.spines[loc].set_visible(False)

    def setup_animation(self):
        """
        Setup animation for AI-AI games.
        Creates an animation that updates every 500ms.
        """
        max_frames = self.game.w * self.game.h
        self.anim = FuncAnimation(
            self.fig, 
            self.draw_move, 
            frames=self.move_generator,
            interval=250,  # 500ms between moves
            repeat=False,
            save_count=max_frames,
            cache_frame_data=False
        )

    def move_generator(self):
        """
        Generator for AI moves in AI-AI games.
        
        Yields:
            tuple: (x, y) coordinates of each move
        """
        score = None
        while score is None:
            self.player = self.game.player
            # Get move from appropriate AI player
            if self.game.player == 1:
                loc = self.player1(self.game)
            else:
                loc = self.player2(self.game)
                
            success = self.game.move(loc)
            if success:
                score = self.game.score
                yield loc
                
    def draw_move(self, move=None):
        """
        Draw a move on the board.
        
        Args:
            move: (x, y) coordinates of the move to draw
                If None, uses last move from game state
        """
        if self.end:
            return
        
        i, j = self.game.last_move if move is None else move
        # Add 0.5 to center the pieces in the grid cells
        i, j = i + 0.5, j + 0.5
        # Color based on player (red for player 1, blue for player 2)
        c = 'salmon' if self.player == 1 else 'lightskyblue'
        self.ax.scatter(i, j, s=800, marker='o', zorder=3, c=c)
        score = self.game.score
        self.draw_winner(score)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def draw_winner(self, score):
        if score is None:
            return
        
        # Draw stars on winning moves
        if score == -1 or score == 1:
            locs = self.game.get_winning_loc()
            c = 'darkred' if score == 1 else 'darkblue'
            # Add 0.5 to center the stars
            adjusted_locs = locs + 0.5
            self.ax.scatter(adjusted_locs[:, 0], adjusted_locs[:, 1], 
                        s=500, marker='*', c=c, zorder=4)
            
            # Update title with winner information
            winner = "Red" if score == 1 else "Blue"
            if self.mode_name:
                # Extract the player type (Human/AI/Random) for the winner
                if score == 1:  # Red wins
                    winner_type = self.mode_name.split("Red: ")[1].split(" vs")[0]
                else:  # Blue wins
                    winner_type = self.mode_name.split("Blue: ")[1]
                self.title.set_text(f"{self.mode_name}\n{winner} ({winner_type}) Wins!")
        else:
            # Draw information
            if self.mode_name:
                self.title.set_text(f"{self.mode_name}\nDraw!")

        # Cleanup click handler at game end
        if self.click_cid:
            self.fig.canvas.mpl_disconnect(self.click_cid)
            self.click_cid = None

        self.end = True
        self.fig.canvas.draw_idle()
        
    def click(self, event):
        """
        Handle mouse clicks for human moves.
        
        Args:
            event: Matplotlib mouse event
        """
        # First check if click is valid and game is still active
        if event.inaxes != self.ax or self.end:
            return
            
        try:
            # Convert click coordinates to board position
            # Subtract 0.5 to compensate for the centered grid
            x = event.xdata - 0.5
            y = event.ydata - 0.5
            loc = (int(round(x)), int(round(y)))
            if not (0 <= loc[0] < self.game.w and 0 <= loc[1] < self.game.h):
                return
                
            self.player = self.game.player
            succeed = self.game.move(loc)

            if succeed:
                self.draw_move()
                
                # Check if game ended after human move
                if self.game.score is not None:
                    return
                
                # Only proceed with AI move if game is not over
                if (self.player1 is not None or self.player2 is not None) and not self.end:
                    succeed = False
                    self.player = self.game.player
                    while not succeed:
                        if self.game.player == 1:
                            loc = self.player1(self.game)
                        else:
                            loc = self.player2(self.game)
                        succeed = self.game.move(loc)
                
                    self.draw_move()
                    
        except Exception as e:
            print(f"Error handling click: {str(e)}")