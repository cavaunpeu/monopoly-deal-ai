import React from "react";
import { CardModel, getCardKind, CashCardModel, PropertyCardModel, RentCardModel, SpecialCardModel } from "@/types/cards";
import { CashCard } from "./CashCard";
import { PropertyCard } from "./PropertyCard";
import { RentCard } from "./RentCard";
import { SpecialCard } from "./SpecialCard";

type Props = {
  card: CardModel;
  count?: number; // For property cards, the current count of this property type
  showProgress?: boolean; // Whether to show progress format for property cards
};

function CardComponent({ card, count, showProgress }: Props) {
  const cardKind = getCardKind(card);
  switch (cardKind) {
    case "CASH":
      return <CashCard card={card as CashCardModel} count={count} />;
    case "PROPERTY":
      return <PropertyCard card={card as PropertyCardModel} count={count} showProgress={showProgress} />;
    case "RENT":
      return <RentCard card={card as RentCardModel} count={count} />;
    case "SPECIAL":
      return <SpecialCard card={card as SpecialCardModel} count={count} />;
    default:
      return (
        <div className="bg-gray-500 text-white rounded-lg p-4 shadow-lg border min-w-[120px] h-[160px] flex items-center justify-center">
          <div className="text-center">
            <div className="text-sm">Unknown Card</div>
            <div className="text-xs opacity-80 mt-1">{JSON.stringify(card)}</div>
          </div>
        </div>
      );
  }
}

export const Card = React.memo(CardComponent);
