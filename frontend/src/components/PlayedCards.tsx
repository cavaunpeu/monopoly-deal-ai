import { CardModel } from "@/types/cards";
import { Card } from "./cards/Card";

type Props = {
  playedCards: CardModel[];
  isHumanTurn: boolean;
};

export function PlayedCards({ playedCards, isHumanTurn }: Props) {
  if (playedCards.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center gap-1">
      <span className="text-xs text-gray-500 mr-2">
        {isHumanTurn ? "You played:" : "AI played:"}
      </span>
      <div className="flex gap-1">
        {playedCards.map((card, index) => (
          <div
            key={index}
            className="transform scale-50 origin-center"
            style={{ width: '60px', height: '80px' }}
          >
            <Card card={card} />
          </div>
        ))}
      </div>
    </div>
  );
}
